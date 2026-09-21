import runpy
import unittest
from pathlib import Path

from probegrpo.envs.public_sokoban import (
    decode_public_task,
    encode_public_task,
    parse_ragen_board,
    shortest_solution_length,
)
from probegrpo.envs.sokoban import SokobanEnv

CONVERT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "prepare_public_sokoban_data.py")
)["convert"]

PUBLIC_PROMPT = """
[Cumulative Observations]:
 # \t # \t # \t # \t # \t # \t
 # \t _ \t O \t X \t P \t # \t
 # \t _ \t _ \t _ \t _ \t # \t
 # \t # \t # \t # \t # \t # \t
 # \t # \t # \t # \t # \t # \t
 # \t # \t # \t # \t # \t # \t
Decide the next action:
"""


class PublicSokobanTest(unittest.TestCase):
    def test_public_prompt_round_trip_and_replay(self):
        board = parse_ragen_board(PUBLIC_PROMPT)
        self.assertEqual(board[1], "# .$@#")
        task_id = encode_public_task(board)
        self.assertEqual(decode_public_task(task_id), board)
        self.assertEqual(shortest_solution_length(board), 1)
        first = SokobanEnv(max_steps=12).replay(task_id, 20000, ("left",))
        second = SokobanEnv(max_steps=12).replay(task_id, 20000, ("left",))
        self.assertEqual(first, second)
        self.assertEqual(first.reward, 1.0)

    def test_malformed_or_non_public_task_fails(self):
        with self.assertRaises(ValueError):
            parse_ragen_board("no board")
        with self.assertRaises(KeyError):
            decode_public_task("private-task")

    def test_public_split_conversion_deduplicates_by_source_index(self):
        records = [
            {
                "prompt": [{"content": PUBLIC_PROMPT}],
                "extra_info": {"index": index},
            }
            for index in (20001, 20000)
        ]
        rows = CONVERT(records, "test", 1, 12)
        self.assertEqual(rows[0]["extra_info"]["source_index"], 20000)
        self.assertEqual(rows[0]["extra_info"]["oracle_shortest_steps"], 1)
        task_id = rows[0]["extra_info"]["task_id"]
        with self.assertRaisesRegex(ValueError, "requested 1 unique"):
            CONVERT(records, "test", 1, 12, excluded_task_ids={task_id})


if __name__ == "__main__":
    unittest.main()
