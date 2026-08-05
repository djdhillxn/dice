"""DiceDial environment built as a thin extension of Isaac Lab's Shadow Hand task.

Key differences from the original design
-----------------------------------------
* **Observation (171-dim)**: the base 157-dim proprioceptive observation is
  extended with (a) the six-way target one-hot, (b) the normalised hold
  progress, (c) the *commanded face normal expressed in world coordinates*
  (3-dim), (d) the *current top-face normal expressed in world coordinates*
  (3-dim), and (e) their dot product (1-dim, the alignment scalar seen by the
  reward).  Items (c-e) give the actor a yaw-invariant, semantically clean
  signal that is fully consistent with the reward function.

* **Stable-gated wrong-face penalty**: the ``wrong_face_penalty_scale`` term
  now fires only when the die is *not* moving fast (angular speed ≤
  ``success_angular_speed``).  This prevents the policy from being punished
  for legitimately rotating through intermediate faces on its way to the
  commanded face.

* **Mild angular-speed penalty at init**: ``angular_speed_scale`` starts at
  -0.02 (set by the ACL Level 0 config).  The ACL manager stiffens it to
  -0.08 at Level 3, so the speed penalty increases only after the policy is
  already rotating effectively.

* **``set_curriculum_level()``**: the AclCurriculum callback calls this method
  mid-training to ratchet the success thresholds without restarting the env.

* **Domain randomisation**: mass ±20 % and friction ±20 % are applied at
  every reset via the event system in the config — training is always in-
  distribution for a range of object properties.
"""

import math

import torch
import torch.nn.functional as functional

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv

from dicedial.geometry import (
    FACE_NORMALS,
    canonical_goal_quaternion,
    rotate_vector_wxyz,
    target_face_alignment,
    top_face,
)


class DiceDialEnv(InHandManipulationEnv):
    """Rotate a numbered die until a commanded face points upward.

    The inherited environment owns simulation setup, the Shadow Hand action
    controller, object and hand resets, fingertip observations, and vectorised
    stepping.  This class adds semantic face commands, continuous command
    switching after a held success, and the supplementary yaw-invariant
    observations.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.target_faces = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.target_one_hot = torch.zeros(self.num_envs, 6, device=self.device)
        self.hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.command_age_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_success_latency = torch.zeros(self.num_envs, device=self.device)
        self.last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_out_of_reach = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_alignment = torch.zeros(self.num_envs, device=self.device)
        self.last_position_error = torch.zeros(self.num_envs, device=self.device)
        self.last_top_face = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.last_completed_face = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.sequence_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Supplementary face-normal observations (updated each step in _get_rewards).
        self._commanded_face_normal_world = torch.zeros(self.num_envs, 3, device=self.device)
        self._top_face_normal_world = torch.zeros(self.num_envs, 3, device=self.device)
        self._alignment_scalar = torch.zeros(self.num_envs, 1, device=self.device)

        all_envs = torch.arange(self.num_envs, device=self.device)
        self._assign_targets(all_envs, advance=False)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def compute_full_observations(self):
        """Return 171-dim actor observation.

        Layout
        ------
        [0:157]   inherited Shadow Hand proprioceptive observation
        [157:163] target face one-hot (6)
        [163:164] normalised hold progress (1)
        [164:167] commanded face normal in world frame (3)
        [167:170] current top-face normal in world frame (3)
        [170:171] alignment scalar — dot(commanded_normal_world, world_up) (1)
        """
        base_obs = super().compute_full_observations()
        hold_fraction = (
            self.hold_counter.float() / max(float(self.cfg.hold_steps), 1.0)
        ).clamp(0.0, 1.0)
        return torch.cat(
            (
                base_obs,
                self.target_one_hot,
                hold_fraction.unsqueeze(-1),
                self._commanded_face_normal_world,
                self._top_face_normal_world,
                self._alignment_scalar,
            ),
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _get_rewards(self):  # noqa: PLR0914
        self.last_success.zero_()
        self.command_age_steps += 1

        alignment = target_face_alignment(self.object_rot, self.target_faces)
        position_error = torch.linalg.vector_norm(self.object_pos - self.in_hand_pos, dim=-1)
        angular_speed = torch.linalg.vector_norm(self.object_angvel, dim=-1)
        current_top_face = top_face(self.object_rot)

        # --- Supplementary observation tensors (shared with compute_full_observations) ---
        face_normals_local = FACE_NORMALS.to(self.device)[self.target_faces.long() - 1]
        self._commanded_face_normal_world = rotate_vector_wxyz(self.object_rot, face_normals_local)
        top_normals_local = FACE_NORMALS.to(self.device)[current_top_face.long() - 1]
        self._top_face_normal_world = rotate_vector_wxyz(self.object_rot, top_normals_local)
        self._alignment_scalar = alignment.unsqueeze(-1)

        # --- Reward components ---
        normalized_alignment = ((alignment + 1.0) * 0.5).clamp(0.0, 1.0)
        alignment_reward = self.cfg.alignment_scale * normalized_alignment.pow(self.cfg.alignment_power)
        position_reward = self.cfg.position_error_scale * position_error
        angular_reward = self.cfg.angular_speed_scale * angular_speed
        action_penalty = self.cfg.action_penalty_scale * torch.sum(self.actions.square(), dim=-1)

        # Stable-gated wrong-face penalty: only penalise when the die is not
        # actively spinning.  This avoids punishing legitimate rotation through
        # intermediate faces on the way to the target.
        die_is_settled = angular_speed <= self.cfg.success_angular_speed
        wrong_face = (current_top_face != self.target_faces) & die_is_settled
        wrong_face_penalty = self.cfg.wrong_face_penalty_scale * wrong_face.float()

        # Hold gate
        angle_threshold = math.cos(math.radians(self.cfg.success_angle_deg))
        hold_condition = (
            (alignment >= angle_threshold)
            & (position_error <= self.cfg.success_position_tolerance)
            & (angular_speed <= self.cfg.success_angular_speed)
        )
        self.hold_counter = torch.where(
            hold_condition,
            self.hold_counter + 1,
            torch.zeros_like(self.hold_counter),
        )
        hold_fraction = (
            self.hold_counter.float() / max(float(self.cfg.hold_steps), 1.0)
        ).clamp(0.0, 1.0)
        hold_reward = self.cfg.hold_bonus_scale * hold_fraction

        success = self.hold_counter >= self.cfg.hold_steps
        success_reward = self.cfg.success_bonus * success.float()

        reward = (
            alignment_reward
            + position_reward
            + angular_reward
            + action_penalty
            + wrong_face_penalty
            + hold_reward
            + success_reward
        )

        self.last_alignment.copy_(alignment)
        self.last_position_error.copy_(position_error)
        self.last_top_face.copy_(current_top_face)

        if success.any():
            success_ids = success.nonzero(as_tuple=False).squeeze(-1)
            self.last_success[success_ids] = True
            self.last_completed_face[success_ids] = self.target_faces[success_ids]
            self.last_success_latency[success_ids] = self.command_age_steps[success_ids].float()
            self.successes[success_ids] += 1

            if self.cfg.switch_target_on_success:
                self._assign_targets(success_ids, advance=True)
                self.last_alignment[success_ids] = target_face_alignment(
                    self.object_rot[success_ids], self.target_faces[success_ids]
                )
            else:
                self.hold_counter[success_ids] = 0
                self.command_age_steps[success_ids] = 0

        log = self.extras.setdefault("log", {})
        log["DiceDial/alignment"] = alignment.mean()
        log["DiceDial/position_error"] = position_error.mean()
        log["DiceDial/angular_speed"] = angular_speed.mean()
        log["DiceDial/hold_fraction"] = hold_fraction.mean()
        log["DiceDial/success_rate_per_step"] = success.float().mean()
        log["DiceDial/commands_per_episode_mean"] = self.successes.float().mean()
        log["DiceDial/drop_rate_per_step"] = self.last_out_of_reach.float().mean()
        # ACL progress signal — consumed by AclCurriculum.
        log["DiceDial/acl_signal"] = self.successes.float().mean()

        # Per-environment fields for evaluation and video metadata.
        reported_hold_fraction = (
            self.hold_counter.float() / max(float(self.cfg.hold_steps), 1.0)
        ).clamp(0.0, 1.0)
        self.extras["dicedial_success"] = self.last_success.clone()
        self.extras["dicedial_completed_face"] = self.last_completed_face.clone()
        self.extras["dicedial_success_latency_steps"] = self.last_success_latency.clone()
        self.extras["dicedial_target_face"] = self.target_faces.clone()
        self.extras["dicedial_top_face"] = self.last_top_face.clone()
        self.extras["dicedial_alignment"] = self.last_alignment.clone()
        self.extras["dicedial_hold_progress"] = reported_hold_fraction.clone()
        self.extras["dicedial_commands_completed"] = self.successes.clone()

        return reward

    # ------------------------------------------------------------------
    # Done / reset
    # ------------------------------------------------------------------

    def _get_dones(self):
        self._compute_intermediate_values()
        out_of_reach = torch.linalg.vector_norm(self.object_pos - self.in_hand_pos, dim=-1) >= self.cfg.fall_dist
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.max_commands_per_episode > 0:
            command_limit = self.successes >= self.cfg.max_commands_per_episode
        else:
            command_limit = torch.zeros_like(out_of_reach)

        self.last_out_of_reach.copy_(out_of_reach)
        self.extras["dicedial_drop"] = out_of_reach.clone()
        self.extras["dicedial_commands_completed"] = self.successes.clone()
        return out_of_reach | command_limit, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        self.hold_counter[env_ids] = 0
        self.command_age_steps[env_ids] = 0
        self.last_success_latency[env_ids] = 0.0
        self.last_success[env_ids] = False
        self.last_out_of_reach[env_ids] = False
        self.last_completed_face[env_ids] = 0
        self.sequence_index[env_ids] = 0

    def _reset_target_pose(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._assign_targets(env_ids, advance=False)

    # ------------------------------------------------------------------
    # Target assignment
    # ------------------------------------------------------------------

    def _assign_targets(self, env_ids, advance):
        if env_ids.numel() == 0:
            return

        mode = self.cfg.target_mode.lower()
        current = self.target_faces[env_ids]

        if mode == "fixed":
            targets = torch.full_like(current, int(self.cfg.fixed_target_face))
        elif mode == "cycle":
            sequence = torch.tensor(self.cfg.target_sequence, dtype=torch.long, device=self.device)
            if advance:
                self.sequence_index[env_ids] = (self.sequence_index[env_ids] + 1) % sequence.numel()
            else:
                self.sequence_index[env_ids] = 0
            targets = sequence[self.sequence_index[env_ids]]
        elif mode == "random":
            targets = torch.randint(1, 7, current.shape, device=self.device)
            if advance:
                duplicate = targets == current
                targets[duplicate] = targets[duplicate] % 6 + 1
        else:
            raise ValueError("target_mode must be one of: fixed, random, cycle")

        self.target_faces[env_ids] = targets
        self.target_one_hot[env_ids] = functional.one_hot(targets - 1, num_classes=6).float()
        self.goal_rot[env_ids] = canonical_goal_quaternion(targets, device=self.device)
        self.hold_counter[env_ids] = 0
        self.command_age_steps[env_ids] = 0

        goal_positions = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_positions, self.goal_rot)
        self.reset_goal_buf[env_ids] = False

    # ------------------------------------------------------------------
    # ACL curriculum hook
    # ------------------------------------------------------------------

    def set_curriculum_level(self, level: dict) -> None:
        """Update success-gate and penalty parameters in place.

        Called by :class:`dicedial.curriculum.AclCurriculum` when the policy's
        rolling mean ``commands_per_episode_mean`` crosses the next threshold.
        No environment restart is required.

        Parameters
        ----------
        level:
            Dict with keys ``success_angle_deg``, ``hold_steps``,
            ``success_angular_speed``, ``angular_speed_scale``.
        """
        self.cfg.success_angle_deg = float(level["success_angle_deg"])
        self.cfg.hold_steps = int(level["hold_steps"])
        self.cfg.success_angular_speed = float(level["success_angular_speed"])
        self.cfg.angular_speed_scale = float(level["angular_speed_scale"])

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_task_metrics(self):
        """Return tensors used by callbacks, evaluation, and video overlays."""
        return {
            "target_face": self.target_faces,
            "top_face": self.last_top_face,
            "alignment": self.last_alignment,
            "position_error": self.last_position_error,
            "hold_progress": (
                self.hold_counter.float() / max(float(self.cfg.hold_steps), 1.0)
            ).clamp(0.0, 1.0),
            "command_age_steps": self.command_age_steps,
            "commands_completed": self.successes,
            "last_success": self.last_success,
            "last_completed_face": self.last_completed_face,
            "last_success_latency_steps": self.last_success_latency,
            "out_of_reach": self.last_out_of_reach,
        }
