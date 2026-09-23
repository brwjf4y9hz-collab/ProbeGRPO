#!/usr/bin/env python3
"""Build fixed ProbeGRPO parquet splits from the public RAGEN Sokoban release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from probegrpo.envs.public_sokoban import (
    PUBLIC_DATASET_REPO,
    PUBLIC_DATASET_REVISION,
    PUBLIC_FILES,
    encode_public_task,
    parse_ragen_board,
    shortest_solution_length,
)
from probegrpo.envs.sokoban import SokobanEnv, initial_messages


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file(split: str, source_dir: Path | None) -> Path:
    filename, expected = PUBLIC_FILES[split]
    if source_dir is None:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=PUBLIC_DATASET_REPO,
                repo_type="dataset",
                revision=PUBLIC_DATASET_REVISION,
                filename=filename,
            )
        )
    else:
        path = source_dir / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path}: expected {expected}, got {actual}")
    return path


def convert(
    records,
    split: str,
    limit: int,
    max_turns: int,
    *,
    excluded_task_ids: set[str] | None = None,
) -> list[dict]:
    excluded_task_ids = excluded_task_ids or set()
    result = []
    seen = set(excluded_task_ids)
    for row in sorted(records, key=lambda item: int(item["extra_info"]["index"])):
        source_index = int(row["extra_info"]["index"])
        content = row["prompt"][0]["content"]
        board = parse_ragen_board(content)
        task_id = encode_public_task(board)
        if task_id in seen:
            continue
        seen.add(task_id)
        oracle_steps = shortest_solution_length(board)
        if oracle_steps is None or oracle_steps > max_turns:
            raise ValueError(
                f"public board {source_index} is not solvable within max_turns={max_turns}"
            )
        state = SokobanEnv(max_steps=max_turns).reset(task_id, source_index)
        result.append(
            {
                "data_source": "probegrpo_ragen_public_sokoban",
                "agent_name": "probegrpo_sokoban",
                "prompt": initial_messages(state),
                "ability": "sokoban",
                "reward_model": {"style": "rule", "ground_truth": "environment_success"},
                "extra_info": {
                    "task_id": task_id,
                    "env_seed": source_index,
                    "max_turns": max_turns,
                    "split": split,
                    "source_index": source_index,
                    "source_repo": PUBLIC_DATASET_REPO,
                    "source_revision": PUBLIC_DATASET_REVISION,
                    "oracle_shortest_steps": oracle_steps,
                },
            }
        )
        if len(result) == limit:
            break
    if len(result) != limit:
        raise ValueError(f"requested {limit} unique {split} boards, found {len(result)}")
    return result


def main() -> None:
    from datasets import Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--test-limit", type=int, default=128)
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args()
    if min(args.train_limit, args.test_limit, args.max_turns) <= 0:
        raise ValueError("limits and max-turns must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_repo": PUBLIC_DATASET_REPO,
        "source_revision": PUBLIC_DATASET_REVISION,
        "source_files": {},
        "derived_splits": {},
        "selection": (
            "lowest unique source indices; test boards overlapping the selected train subset "
            "are excluded"
        ),
        "max_turns": args.max_turns,
    }
    selected_task_ids: set[str] = set()
    for split, limit in (("train", args.train_limit), ("test", args.test_limit)):
        path = source_file(split, args.source_dir)
        source = Dataset.from_parquet(str(path))
        rows = convert(
            source,
            split,
            limit,
            args.max_turns,
            excluded_task_ids=selected_task_ids if split == "test" else None,
        )
        selected_task_ids.update(row["extra_info"]["task_id"] for row in rows)
        output = args.output_dir / f"{split}.parquet"
        Dataset.from_list(rows).to_parquet(str(output))
        manifest["source_files"][split] = {
            "path": PUBLIC_FILES[split][0],
            "sha256": PUBLIC_FILES[split][1],
            "total_rows": len(source),
        }
        manifest["derived_splits"][split] = {
            "rows": len(rows),
            "source_index_min": min(row["extra_info"]["source_index"] for row in rows),
            "source_index_max": max(row["extra_info"]["source_index"] for row in rows),
            "sha256": sha256(output),
            "oracle_shortest_steps_max": max(
                row["extra_info"]["oracle_shortest_steps"] for row in rows
            ),
        }
    train_ids = {
        row["extra_info"]["task_id"]
        for row in Dataset.from_parquet(str(args.output_dir / "train.parquet"))
    }
    test_ids = {
        row["extra_info"]["task_id"]
        for row in Dataset.from_parquet(str(args.output_dir / "test.parquet"))
    }
    if train_ids & test_ids:
        raise ValueError("public train/test board overlap detected in selected subsets")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared public RAGEN Sokoban: train={args.train_limit}, "
        f"test={args.test_limit}, revision={PUBLIC_DATASET_REVISION[:7]}"
    )


if __name__ == "__main__":
    main()
