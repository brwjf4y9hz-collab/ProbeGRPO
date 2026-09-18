"""A dependency-free Sokoban micro-environment for replay and CI tests."""

from __future__ import annotations

from typing import Dict, Tuple

from ..replay import ReplayableEnv, canonical_state_hash
from ..types import ReplayState, RolloutOutcome

Position = Tuple[int, int]


class TinySokobanEnv(ReplayableEnv):
    """One-box Sokoban with deterministic transitions and a short horizon.

    The default task starts with the box immediately left of the player and the target immediately
    left of the box. Pushing left solves the task; moving right consumes the short horizon.
    """

    ACTIONS: Dict[str, Position] = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(self, max_steps: int = 2) -> None:
        self.max_steps = max_steps
        self.task_id = "tiny-0"
        self.seed = 0
        self.player: Position = (1, 4)
        self.box: Position = (1, 3)
        self.target: Position = (1, 2)
        self.steps = 0
        self.done = False
        self.reward = 0.0

    def reset(self, task_id: str, seed: int) -> ReplayState:
        if task_id not in {"tiny-0", "tiny-replay"}:
            raise KeyError(f"Unknown tiny Sokoban task: {task_id}")
        self.task_id = task_id
        self.seed = int(seed)
        self.player = (1, 4)
        self.box = (1, 3)
        self.target = (1, 2)
        self.steps = 0
        self.done = False
        self.reward = 0.0
        return self._state()

    def step(self, action: str) -> ReplayState:
        if self.done:
            raise RuntimeError("Cannot step a terminal TinySokobanEnv")
        self.steps += 1
        delta = self.ACTIONS.get(action)
        if delta is not None:
            destination = self._add(self.player, delta)
            if destination == self.box:
                box_destination = self._add(self.box, delta)
                if self._is_floor(box_destination):
                    self.box = box_destination
                    self.player = destination
            elif self._is_floor(destination):
                self.player = destination

        if self.box == self.target:
            self.reward = 1.0
            self.done = True
        elif self.steps >= self.max_steps:
            self.done = True
        return self._state()

    def _state(self) -> ReplayState:
        raw = {
            "task_id": self.task_id,
            "seed": self.seed,
            "player": self.player,
            "box": self.box,
            "target": self.target,
            "steps": self.steps,
            "done": self.done,
        }
        return ReplayState(
            observation=self._render(),
            state_hash=canonical_state_hash(raw),
            legal_actions=self._legal_actions(),
            terminal=self.done,
            reward=self.reward,
            metadata=raw,
        )

    def _legal_actions(self) -> Tuple[str, ...]:
        if self.done:
            return ()
        legal = []
        for action, delta in self.ACTIONS.items():
            destination = self._add(self.player, delta)
            if destination == self.box:
                if self._is_floor(self._add(self.box, delta)):
                    legal.append(action)
            elif self._is_floor(destination):
                legal.append(action)
        return tuple(sorted(legal))

    @staticmethod
    def _add(left: Position, right: Position) -> Position:
        return (left[0] + right[0], left[1] + right[1])

    @staticmethod
    def _is_floor(position: Position) -> bool:
        row, col = position
        return row == 1 and 1 <= col <= 5

    def _render(self) -> str:
        cells = ["#", " ", " ", " ", " ", " ", "#"]
        target_col = self.target[1]
        box_col = self.box[1]
        player_col = self.player[1]
        cells[target_col] = "."
        cells[box_col] = "*" if self.box == self.target else "B"
        cells[player_col] = "P"
        return "#######\n" + "".join(cells) + "\n#######"


class TinySokobanSuffixPolicy:
    """Deterministic suffix runner used by tests and the local demo."""

    def __call__(
        self,
        env: ReplayableEnv,
        anchor_state: ReplayState,
        forced_action: str,
        sampling_seed: int,
    ) -> RolloutOutcome:
        del anchor_state, sampling_seed
        actions = [forced_action]
        token_count = _action_tokens(forced_action)
        state = env.step(forced_action)
        while not state.terminal:
            action = "left" if "left" in state.legal_actions else state.legal_actions[0]
            actions.append(action)
            token_count += _action_tokens(action)
            state = env.step(action)
        return RolloutOutcome(
            reward=state.reward,
            token_count=token_count,
            steps=len(actions),
            terminal=state.terminal,
            actions=tuple(actions),
        )


def _action_tokens(action: str) -> int:
    return max(1, len(action.split()))
