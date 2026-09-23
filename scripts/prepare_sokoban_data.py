"""Create tiny Sokoban fixture parquet files for integration smoke tests."""

import argparse
from pathlib import Path

from probegrpo.envs.sokoban import EVAL_TASKS, TRAIN_TASKS, SokobanEnv, initial_messages


def rows(tasks, split):
    return [
        {
            "data_source": "probegrpo_sokoban_fixtures",
            "agent_name": "probegrpo_sokoban",
            "prompt": initial_messages(SokobanEnv().reset(task, 17)),
            "ability": "sokoban",
            "reward_model": {"style": "rule", "ground_truth": "environment_success"},
            "extra_info": {
                "task_id": task,
                "env_seed": 17,
                "max_turns": 8,
                "split": split,
                "index": i,
            },
        }
        for i, task in enumerate(tasks)
    ]


def main():
    from datasets import Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, tasks in (("train", TRAIN_TASKS), ("test", EVAL_TASKS)):
        Dataset.from_list(rows(tasks, split)).to_parquet(str(args.output_dir / f"{split}.parquet"))
    print("Wrote 2 train / 1 held-out fixture tasks; not a benchmark dataset.")


if __name__ == "__main__":
    main()
