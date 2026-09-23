"""Install the ProbeGRPO advantage call in the pinned verl v1 trainer.

The project keeps the external verl checkout separate. This small, idempotent patch
adds the call immediately after standard GRPO advantage calculation and before the
trainer writes nested advantages to TransferQueue.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

VERL_COMMIT = "cf14ded3a448107e70a206fd201817cc1cbae348"
TARGET = Path("verl/trainer/ppo/v1/trainer_base.py")
MARKER = "        # 4. write nested advantages and returns back to TransferQueue\n"
INJECTION = '''        probe_config = self.config.get("probe", {})
        if probe_config.get("enabled", False):
            from pathlib import Path as _ProbePath

            from probegrpo.integration.verl_v1_sidecar import apply_sidecar_probe_credits

            data, probe_metrics = apply_sidecar_probe_credits(
                data,
                batch.keys,
                sidecar_dir=_ProbePath(self.config.trainer.rollout_data_dir)
                / "episodes"
                / f"step-{self.global_steps}",
                budget=int(probe_config.get("budget", 1)),
                lambda_coef=float(probe_config.get("lambda_coef", 0.5)),
            )
            metrics.update(probe_metrics)
'''


def install(verl_dir: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(verl_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != VERL_COMMIT:
        raise RuntimeError(f"expected pinned verl {VERL_COMMIT}, found {revision}")
    target = verl_dir / TARGET
    original = target.read_text()
    if INJECTION in original:
        return "already installed"
    if original.count(MARKER) != 1:
        raise RuntimeError("the expected verl advantage writeback marker was not found once")
    diff = subprocess.check_output(
        ["git", "-C", str(verl_dir), "diff", "--", str(TARGET)], text=True
    )
    if diff:
        raise RuntimeError(f"refusing to modify an already edited trainer: {target}")
    backup = target.with_suffix(".py.probegrpo.backup")
    if backup.exists():
        raise RuntimeError(f"unexpected existing backup: {backup}")
    backup.write_text(original)
    target.write_text(original.replace(MARKER, INJECTION + MARKER))
    return f"installed; original saved at {backup}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("verl_dir", type=Path)
    args = parser.parse_args()
    print(install(args.verl_dir.resolve()))


if __name__ == "__main__":
    main()
