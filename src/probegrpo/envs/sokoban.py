"""Small two-dimensional Sokoban fixtures; not a published benchmark distribution."""

from __future__ import annotations

from ..replay import ReplayableEnv, canonical_state_hash
from ..types import ReplayState
from .public_sokoban import decode_public_task

LEVELS = {
    "push-up": ("#######", "#  .  #", "#  $  #", "#     #", "#  @  #", "#######"),
    "push-right": ("#######", "#     #", "#@ $ .#", "#     #", "#######"),
    "push-left": ("#######", "#     #", "#. $ @#", "#     #", "#######"),
}
TRAIN_TASKS = ("push-up", "push-right")
EVAL_TASKS = ("push-left",)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


class SokobanEnv(ReplayableEnv):
    """One box, immutable walls/goals, sparse success reward, bounded action attempts.

    Invalid text or blocked moves consume one step. State hashes include the complete
    transition state, level identity and horizon, not just the rendered observation.
    """

    def __init__(self, max_steps: int = 8):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self._ready = False

    def reset(self, task_id: str, seed: int) -> ReplayState:
        board = LEVELS[task_id] if task_id in LEVELS else decode_public_task(task_id)
        self.task_id, self.seed = task_id, int(seed)
        self.board = board
        self.floor = {(r, c) for r, row in enumerate(board) for c, x in enumerate(row) if x != "#"}
        self.goals = {
            (r, c) for r, row in enumerate(board) for c, x in enumerate(row) if x in ".*+"
        }
        self.boxes = {(r, c) for r, row in enumerate(board) for c, x in enumerate(row) if x in "$*"}
        self.player = next(
            (r, c) for r, row in enumerate(board) for c, x in enumerate(row) if x in "@+"
        )
        self.steps, self.invalid_actions = 0, 0
        self._ready = True
        return self.state()

    def _destination(self, action):
        dr, dc = ACTIONS[action]
        r, c = self.player
        return (r + dr, c + dc), (r + 2 * dr, c + 2 * dc)

    def _legal(self):
        actions = []
        for action in ACTIONS:
            dest, beyond = self._destination(action)
            if dest in self.floor and (
                dest not in self.boxes or (beyond in self.floor and beyond not in self.boxes)
            ):
                actions.append(action)
        return tuple(actions)

    def step(self, action: str) -> ReplayState:
        state = self.state()
        if state.terminal:
            raise RuntimeError("Cannot step terminal SokobanEnv")
        valid = action in state.legal_actions
        self.steps += 1
        if valid:
            dest, beyond = self._destination(action)
            if dest in self.boxes:
                self.boxes.remove(dest)
                self.boxes.add(beyond)
            self.player = dest
        else:
            self.invalid_actions += 1
        return self.state()

    def state(self) -> ReplayState:
        if not self._ready:
            raise RuntimeError("reset must precede state/step")
        success = self.boxes == self.goals
        terminal = success or self.steps >= self.max_steps
        payload = {
            "task_id": self.task_id,
            "seed": self.seed,
            "layout": self.board,
            "player": self.player,
            "boxes": sorted(self.boxes),
            "goals": sorted(self.goals),
            "steps": self.steps,
            "max_steps": self.max_steps,
            "invalid_actions": self.invalid_actions,
        }
        board = []
        for r, row in enumerate(self.board):
            cells = []
            for c, _ in enumerate(row):
                pos = (r, c)
                char = "#" if pos not in self.floor else ("." if pos in self.goals else " ")
                if pos in self.boxes:
                    char = "*" if pos in self.goals else "$"
                if pos == self.player:
                    char = "+" if pos in self.goals else "@"
                cells.append(char)
            board.append("".join(cells))
        return ReplayState(
            observation="\n".join(board),
            state_hash=canonical_state_hash(payload),
            legal_actions=() if terminal else self._legal(),
            terminal=terminal,
            reward=float(success),
            metadata={**payload, "success": success},
        )


def observation_message(state: ReplayState) -> str:
    return (
        f"{state.observation}\nLegal moves: {', '.join(state.legal_actions)}.\n"
        "Reply with exactly one move: up, down, left, or right."
    )


def initial_messages(state: ReplayState) -> list:
    return [
        {
            "role": "system",
            "content": "Solve Sokoban. # wall, @ player, $ box, . goal, "
            "* box on goal, + player on goal. Push the box onto the goal; you cannot pull it. "
            "Output only one move, with no explanation.",
        },
        {"role": "user", "content": observation_message(state)},
    ]
