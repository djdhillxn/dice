"""Pure tensor helpers for the DICE joint-target controller."""

from __future__ import annotations

import torch


def normalize_selected_joint_targets(
    targets: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    joint_indices: list[int] | tuple[int, ...],
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select actuated joints and normalize their targets into ``[-1, 1]``.

    All input tensors are batched as ``[num_envs, num_joints]``. Environment
    rows must be selected before joint columns; reversing that order silently
    indexes the environment dimension with joint identifiers.
    """

    rows = slice(None) if env_ids is None else env_ids
    selected_targets = targets[rows][:, joint_indices]
    selected_lower = lower_limits[rows][:, joint_indices]
    selected_upper = upper_limits[rows][:, joint_indices]
    # Joint limits come from the validated PhysX articulation. Avoid a Python
    # conditional on a CUDA reduction here: this helper runs in the reward hot
    # path, where such a check would synchronize the GPU every control step.
    joint_range = (selected_upper - selected_lower).clamp_min(
        torch.finfo(selected_targets.dtype).eps
    )
    normalized = 2.0 * (selected_targets - selected_lower) / joint_range - 1.0
    return normalized.clamp(-1.0, 1.0)
