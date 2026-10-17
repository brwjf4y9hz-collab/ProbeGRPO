#!/usr/bin/env python3
"""Execution-only profiles and explicitly scoped GPU checks; importable without torch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_NAMES = ("single_48gb", "single_24gb", "dual_24gb")
EXECUTION_KEYS = {
    "trainer.nnodes", "trainer.n_gpus_per_node",
    "actor_rollout_ref.actor.fsdp_config.fsdp_size",
    "actor_rollout_ref.actor.fsdp_config.param_offload",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    "actor_rollout_ref.actor.entropy_from_logits_chunk_size",
    "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "actor_rollout_ref.rollout.gpu_memory_utilization",
    "actor_rollout_ref.rollout.max_num_batched_tokens",
    "actor_rollout_ref.rollout.max_num_seqs",
}


def profile_hash(profile: dict) -> str:
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()


def load_profile(name: str = "single_48gb") -> dict:
    if name not in PROFILE_NAMES:
        raise ValueError(f"unknown hardware profile: {name}")
    profile = json.loads((ROOT / "configs/hardware" / f"{name}.json").read_text())
    if profile["name"] != name or set(profile["hydra_overrides"]) != EXECUTION_KEYS:
        raise ValueError("hardware profiles may only set the declared execution knobs")
    knobs = profile["hydra_overrides"]
    count = profile["gpu_count"]
    if (count not in (1, 2) or knobs["trainer.n_gpus_per_node"] != count
            or knobs["trainer.nnodes"] != 1
            or knobs["actor_rollout_ref.actor.fsdp_config.fsdp_size"] != count
            or knobs["actor_rollout_ref.rollout.tensor_model_parallel_size"] != count):
        raise ValueError("inconsistent GPU partition profile")
    # Public runner permits 1024 prompt + 1024 response positions. Dynamic
    # batching must fit a complete sample; reducing this cap is not safe chunking.
    if (knobs["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] < 2048
            or knobs["actor_rollout_ref.rollout.max_num_batched_tokens"] < 2048):
        raise ValueError("execution token budget must fit a full 1024+1024 sample")
    return profile


def device_environment(base: dict, requested: str | None = None) -> dict:
    """Never expand scheduler-provided CUDA visibility or automatically take all school GPUs."""
    env = dict(base)
    if requested is None:
        return env
    tokens = [token.strip() for token in requested.split(",")]
    if (not tokens or len(set(tokens)) != len(tokens)
            or any(not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", token) for token in tokens)):
        raise ValueError("GPU devices must be distinct assigned indices or GPU UUIDs")
    value = ",".join(tokens)
    if "CUDA_VISIBLE_DEVICES" in env and env["CUDA_VISIBLE_DEVICES"] != value:
        raise ValueError("refusing to override existing/scheduler CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = value
    return env


def validate_inventory(profile: dict, inventory: dict) -> None:
    devices = inventory["devices"]
    if len(devices) != profile["gpu_count"]:
        raise ValueError("visible GPU count must exactly match profile; select only allocated devices")
    for device in devices:
        if device["total_memory_mib"] < profile["min_memory_mib"]:
            raise ValueError("GPU memory is smaller than profile requirement; no automatic downsizing")
        if device["free_memory_mib"] / device["total_memory_mib"] < profile["min_free_fraction"]:
            raise ValueError("GPU is not sufficiently free; obtain an exclusive allocation, do not evict others")


def hardware_record(profile: dict, inventory: dict | None = None) -> dict:
    return {"name": profile["name"], "sha256": profile_hash(profile), "settings": profile,
            "inventory": inventory,
            "gpu_signature": sorted([[d["name"], d["total_memory_mib"]]
                                     for d in (inventory or {}).get("devices", [])])}


def inspect_gpus(profile: dict) -> dict:
    import torch  # Only the future explicitly requested GPU execution path imports torch.

    count = torch.cuda.device_count()
    if count != profile["gpu_count"]:
        raise ValueError(f"expected exactly {profile['gpu_count']} allocated visible GPUs, found {count}")
    devices = []
    for index in range(count):
        prop = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        devices.append({"index": index, "name": prop.name,
                        "total_memory_mib": round(total / 2**20),
                        "free_memory_mib": round(free / 2**20)})
    inventory = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "devices": devices}
    validate_inventory(profile, inventory)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_gpus(load_profile(args.profile))))


if __name__ == "__main__":
    main()
