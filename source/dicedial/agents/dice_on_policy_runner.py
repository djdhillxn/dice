"""DICE-specific RSL-RL runner diagnostics with unchanged PPO behavior."""

from __future__ import annotations

import math

import torch
from rsl_rl.runners import OnPolicyRunner


class DiceOnPolicyRunner(OnPolicyRunner):
    """Add policy-output diagnostics that RSL-RL does not log by default."""

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width=width, pad=pad)
        if self.writer is None:
            return

        policy = self.alg.policy
        was_training = policy.training
        policy.eval()
        try:
            with torch.inference_mode():
                mean_actions = policy.act_inference(locs["obs"])
        finally:
            policy.train(was_training)

        with torch.inference_mode():
            self.writer.add_scalar(
                "Policy/action_mean_abs", mean_actions.abs().mean(), locs["it"]
            )
            self.writer.add_scalar(
                "Policy/action_mean_rms",
                torch.sqrt(mean_actions.square().mean()),
                locs["it"],
            )
            self.writer.add_scalar(
                "Policy/action_mean_out_of_bounds_rate",
                (mean_actions.abs() >= 1.0).float().mean(),
                locs["it"],
            )

            action_std = policy.action_std
            min_noise_std = float(action_std.min().item())
            max_noise_std = float(action_std.max().item())
            self.writer.add_scalar("Policy/min_noise_std", min_noise_std, locs["it"])
            self.writer.add_scalar("Policy/max_noise_std", max_noise_std, locs["it"])

        if (
            not math.isfinite(min_noise_std)
            or not math.isfinite(max_noise_std)
            or min_noise_std <= 0.0
        ):
            raise RuntimeError(
                "DICE policy produced a non-finite or non-positive action noise "
                f"standard deviation (min={min_noise_std}, max={max_noise_std})."
            )
        if max_noise_std > 2.0 and not getattr(self, "_noise_warning_emitted", False):
            print(
                "[DICE WARNING] Policy noise standard deviation exceeded 2.0; "
                "inspect action saturation before continuing a long run.",
                flush=True,
            )
            self._noise_warning_emitted = True
