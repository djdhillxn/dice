"""Automatic Curriculum Learning (ACL) manager for DiceDial.

Strategy
--------
A single ``DiceDial-Shadow-Sequence-v0`` environment is used for the entire
training run.  The policy starts on Level 0 (relaxed success thresholds) and
the curriculum automatically advances when the policy's rolling mean
``commands_per_episode_mean`` exceeds the level-specific threshold.

Level definitions
-----------------
Four levels bridge from easy enough for an untrained policy (Level 0) to the
final evaluation difficulty (Level 3):

  Level 0  success_angle=30°, hold=8  steps, ω_max=2.0  → advance at cpm>0.5
  Level 1  success_angle=24°, hold=12 steps, ω_max=1.6  → advance at cpm>1.0
  Level 2  success_angle=20°, hold=16 steps, ω_max=1.4  → advance at cpm>1.5
  Level 3  success_angle=16°, hold=20 steps, ω_max=1.25 → final (no advance)

``commands_per_episode_mean`` (cpm) is the mean number of distinct face
commands the policy completes per episode across all parallel environments.
It rises from ~0 when the policy drops the die constantly to >2 for a
well-trained policy.

Usage
-----
Instantiate once, then call ``step()`` after every RSL-RL training chunk::

    acl = AclCurriculum(env.unwrapped)
    for chunk in training_chunks:
        runner.learn(chunk)
        acl.step(env.unwrapped.extras.get("log", {}), runner.it)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dicedial.tasks.dice_dial_env import DiceDialEnv

logger = logging.getLogger(__name__)


class AclCurriculum:
    """Rolling-window ACL manager.

    Parameters
    ----------
    env:
        Unwrapped ``DiceDialEnv`` instance.  The curriculum calls
        ``env.set_curriculum_level(level_dict)`` when advancing.
    window:
        Number of ``step()`` calls over which the rolling mean is computed.
        With a 100-iteration chunk, ``window=50`` corresponds to 5,000
        RSL-RL iterations of history.
    """

    #: Four curriculum levels from relaxed (0) to final evaluation (3).
    LEVELS: list[dict] = [
        {
            "name": "Level 0 — relaxed",
            "success_angle_deg": 30.0,
            "hold_steps": 8,
            "success_angular_speed": 2.0,
            "angular_speed_scale": -0.02,
            "advance_threshold": 0.5,   # commands per episode
        },
        {
            "name": "Level 1",
            "success_angle_deg": 24.0,
            "hold_steps": 12,
            "success_angular_speed": 1.6,
            "angular_speed_scale": -0.04,
            "advance_threshold": 1.0,
        },
        {
            "name": "Level 2",
            "success_angle_deg": 20.0,
            "hold_steps": 16,
            "success_angular_speed": 1.4,
            "angular_speed_scale": -0.06,
            "advance_threshold": 1.5,
        },
        {
            "name": "Level 3 — final",
            "success_angle_deg": 16.0,
            "hold_steps": 20,
            "success_angular_speed": 1.25,
            "angular_speed_scale": -0.08,
            "advance_threshold": None,  # terminal level
        },
    ]

    def __init__(self, env: "DiceDialEnv", window: int = 50) -> None:
        self.env = env
        self.window = max(int(window), 1)
        self.current_level_idx = 0
        self._history: list[float] = []

        # Apply initial level immediately.
        self._apply_level(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_level(self) -> dict:
        return self.LEVELS[self.current_level_idx]

    @property
    def at_final_level(self) -> bool:
        return self.current_level_idx >= len(self.LEVELS) - 1

    def step(self, log_dict: dict, iteration: int) -> bool:
        """Check rolling metric and advance if the threshold is met.

        Parameters
        ----------
        log_dict:
            ``env.extras["log"]`` dict from the last RSL-RL rollout.
        iteration:
            Current RSL-RL runner iteration (for logging).

        Returns
        -------
        bool
            ``True`` if the curriculum advanced this call.
        """
        if self.at_final_level:
            return False

        # Extract the ACL signal — commands completed per episode mean.
        raw = log_dict.get("DiceDial/acl_signal", None)
        if raw is None:
            raw = log_dict.get("DiceDial/commands_per_episode_mean", 0.0)

        if hasattr(raw, "item"):
            raw = raw.item()

        self._history.append(float(raw))
        if len(self._history) > self.window:
            self._history.pop(0)

        if len(self._history) < self.window:
            # Not enough data yet.
            return False

        mean_cpm = sum(self._history) / len(self._history)
        threshold = self.current_level["advance_threshold"]

        if threshold is not None and mean_cpm >= threshold:
            self.current_level_idx += 1
            self._apply_level(self.current_level_idx)
            self._history.clear()
            logger.info(
                "[ACL] iter=%d  advanced to %s  "
                "(mean_cpm=%.3f ≥ %.2f)",
                iteration,
                self.current_level["name"],
                mean_cpm,
                threshold,
            )
            return True

        return False

    def state_dict(self) -> dict:
        """Return serialisable state for checkpoint saving."""
        return {
            "current_level_idx": self.current_level_idx,
            "history": list(self._history),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a checkpoint."""
        self.current_level_idx = int(state["current_level_idx"])
        self._history = list(state.get("history", []))
        self._apply_level(self.current_level_idx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_level(self, idx: int) -> None:
        level = self.LEVELS[idx]
        self.env.set_curriculum_level(level)
        logger.info("[ACL] applied %s: angle=%.1f° hold=%d ω_max=%.2f ω_scale=%.3f",
                    level["name"],
                    level["success_angle_deg"],
                    level["hold_steps"],
                    level["success_angular_speed"],
                    level["angular_speed_scale"])
