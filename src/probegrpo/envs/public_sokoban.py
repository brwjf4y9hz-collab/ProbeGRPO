"""Adapters for the immutable public RAGEN Sokoban train/test release."""

from __future__ import annotations

import base64
import re
from collections import deque

PUBLIC_DATASET_REPO = "ZihanWang314/ragen-datasets"
PUBLIC_DATASET_REVISION = "e2060cf7c51da891a68e0e7437309e5ec052e45b"
PUBLIC_FILES = {
    "train": (
        "sokoban/train.parquet",
        "b34ed8c6620df72bd77b0159b2f9957763a8da039614426ad2251bcf98160c1b",
    ),
    "test": (
        "sokoban/test.parquet",
        "558c011757a4a37d8ef166e403008d87cb7caaafa4368405e80c7ef3d2e9a6fc",
    ),
}
PUBLIC_TASK_PREFIX = "ragen-public-v1:"
_SYMBOLS = {"#": "#", "_": " ", "O": ".", "X": "$", "P": "@", "√": "*", "S": "+"}
_BOARD_RE = re.compile(
    r"\[Cumulative Observations\]:\s*(.*?)\s*Decide the next action:",
    flags=re.DOTALL,
)


def parse_ragen_board(prompt: str) -> tuple[str, ...]:
    """Extract and translate RAGEN's tabular 6x6 board from a released prompt."""

    match = _BOARD_RE.search(prompt)
    if match is None:
        raise ValueError("public RAGEN prompt has no cumulative-observation board")
    rows = []
    for raw_line in match.group(1).splitlines():
        cells = [cell.strip() for cell in raw_line.split("\t") if cell.strip()]
        if cells:
            try:
                rows.append("".join(_SYMBOLS[cell] for cell in cells))
            except KeyError as error:
                raise ValueError(f"unsupported public Sokoban symbol: {error.args[0]}") from error
    return validate_public_board(tuple(rows))


def encode_public_task(board: tuple[str, ...]) -> str:
    """Embed a released board in the task ID so replay needs no mutable registry."""

    board = validate_public_board(board)
    payload = "\n".join(board).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return PUBLIC_TASK_PREFIX + encoded


def decode_public_task(task_id: str) -> tuple[str, ...]:
    if not task_id.startswith(PUBLIC_TASK_PREFIX):
        raise KeyError(task_id)
    encoded = task_id[len(PUBLIC_TASK_PREFIX) :]
    try:
        padding = "=" * (-len(encoded) % 4)
        board = tuple(base64.urlsafe_b64decode(encoded + padding).decode("utf-8").splitlines())
    except Exception as error:
        raise ValueError("malformed public Sokoban task ID") from error
    return validate_public_board(board)


def validate_public_board(board: tuple[str, ...]) -> tuple[str, ...]:
    if not board or len({len(row) for row in board}) != 1 or not all(board):
        raise ValueError("public Sokoban board must be a non-empty rectangle")
    if any(char not in "# .$@*+" for row in board for char in row):
        raise ValueError("public Sokoban board contains unsupported characters")
    player_count = sum(row.count("@") + row.count("+") for row in board)
    box_count = sum(row.count("$") + row.count("*") for row in board)
    goal_count = sum(row.count(".") + row.count("*") + row.count("+") for row in board)
    if player_count != 1 or box_count != 1 or goal_count != 1:
        raise ValueError("pilot supports exactly one player, one box, and one goal")
    if any(char != "#" for char in board[0] + board[-1]):
        raise ValueError("public Sokoban board must have a closed boundary")
    if any(row[0] != "#" or row[-1] != "#" for row in board):
        raise ValueError("public Sokoban board must have a closed boundary")
    return board


def shortest_solution_length(board: tuple[str, ...]) -> int | None:
    """Return the exact minimum move count for the supported one-box public boards."""

    board = validate_public_board(board)
    floor = {
        (r, c) for r, row in enumerate(board) for c, char in enumerate(row) if char != "#"
    }
    goal = next(
        (r, c) for r, row in enumerate(board) for c, char in enumerate(row) if char in ".*+"
    )
    box = next((r, c) for r, row in enumerate(board) for c, char in enumerate(row) if char in "$*")
    player = next(
        (r, c) for r, row in enumerate(board) for c, char in enumerate(row) if char in "@+"
    )
    queue = deque([(player, box, 0)])
    seen = {(player, box)}
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while queue:
        player, box, distance = queue.popleft()
        if box == goal:
            return distance
        for dr, dc in moves:
            destination = (player[0] + dr, player[1] + dc)
            if destination not in floor:
                continue
            next_box = box
            if destination == box:
                next_box = (box[0] + dr, box[1] + dc)
                if next_box not in floor:
                    continue
            state = (destination, next_box)
            if state not in seen:
                seen.add(state)
                queue.append((*state, distance + 1))
    return None
