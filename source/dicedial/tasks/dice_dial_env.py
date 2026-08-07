"""Command-conditioned in-hand die reorientation for DICE."""

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


class DiceEnv(InHandManipulationEnv):
    """Rotate a held cube until the requested semantic face points upward.

    A success requires the final 16-degree orientation gate, palm-position
    retention, angular speed at or below 1.25 rad/s, and twenty consecutive
    control steps satisfying all three conditions.  A new face command is then
    assigned without resetting the hand or object.
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

        all_envs = torch.arange(self.num_envs, device=self.device)
        self._assign_targets(all_envs, advance=False)
        self.last_alignment.copy_(target_face_alignment(self.object_rot, self.target_faces))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def compute_full_observations(self):
        """Return stock 157 features plus 17 command and geometric orientation features.

        Features (174 total):
        - Stock Shadow Hand obs: 157
        - Target face one-hot: 6
        - Normalized hold progress: 1
        - Target face alignment: 1
        - Commanded face normal in world frame: 3
        - Current top face normal in world frame: 3
        - Rotation axis error vector (cross product): 3
        """

        base_observation = super().compute_full_observations()
        hold_progress = (
            self.hold_counter.float() / max(float(self.cfg.hold_steps), 1.0)
        ).clamp(0.0, 1.0)
        alignment = target_face_alignment(self.object_rot, self.target_faces)

        face_normals_dev = FACE_NORMALS.to(self.device)
        commanded_local_normals = face_normals_dev[self.target_faces - 1]
        commanded_world_normals = rotate_vector_wxyz(self.object_rot, commanded_local_normals)

        current_top = top_face(self.object_rot)
        top_local_normals = face_normals_dev[current_top - 1]
        top_world_normals = rotate_vector_wxyz(self.object_rot, top_local_normals)

        axis_error = torch.cross(top_world_normals, commanded_world_normals, dim=-1)

        return torch.cat(
            (
                base_observation,
                self.target_one_hot,
                hold_progress.unsqueeze(-1),
                alignment.unsqueeze(-1),
                commanded_world_normals,
                top_world_normals,
                axis_error,
            ),
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _get_rewards(self):
        self.last_success.zero_()
        self.last_completed_face.zero_()
        self.last_success_latency.zero_()
        self.command_age_steps += 1

        alignment = target_face_alignment(self.object_rot, self.target_faces)
        position_error = torch.linalg.vector_norm(self.object_pos - self.in_hand_pos, dim=-1)
        angular_speed = torch.linalg.vector_norm(self.object_angvel, dim=-1)
        if self.cfg.emit_step_metrics:
            current_top_face = top_face(self.object_rot)
        else:
            current_top_face = None
        out_of_reach = position_error >= self.cfg.fall_dist

        # Potential-based progress reward: rewards delta alignment toward target.
        # Loitering statically yields 0 progress reward!
        delta_alignment = alignment - self.last_alignment
        progress_reward = self.cfg.alignment_scale * delta_alignment

        position_penalty = self.cfg.position_error_scale * position_error

        angular_gate = math.cos(math.radians(self.cfg.angular_penalty_gate_deg))
        near_target = alignment >= angular_gate
        angular_penalty = (
            self.cfg.angular_speed_scale
            * angular_speed
            * near_target.float()
        )

        action_penalty = self.cfg.action_penalty_scale * torch.sum(self.actions.square(), dim=-1)
        drop_penalty = self.cfg.drop_penalty * out_of_reach.float()

        success_angle = math.cos(math.radians(self.cfg.success_angle_deg))
        hold_condition = (
            (alignment >= success_angle)
            & (position_error <= self.cfg.success_position_tolerance)
            & (angular_speed <= self.cfg.success_angular_speed)
        )
        self.hold_counter = torch.where(
            hold_condition,
            self.hold_counter + 1,
            torch.zeros_like(self.hold_counter),
        )

        hold_progress = (
            self.hold_counter.float() / max(float(self.cfg.hold_steps), 1.0)
        ).clamp(0.0, 1.0)

        # Continuous hold progress shaping rewards climbing toward the 20-step gate
        hold_shaping = self.cfg.hold_progress_scale * hold_progress * hold_condition.float()

        success = self.hold_counter >= self.cfg.hold_steps
        success_reward = self.cfg.success_bonus * success.float()

        reward = (
            progress_reward
            + hold_shaping
            + position_penalty
            + angular_penalty
            + action_penalty
            + success_reward
            + drop_penalty
        )

        self.last_alignment.copy_(alignment)
        self.last_position_error.copy_(position_error)
        if current_top_face is not None:
            self.last_top_face.copy_(current_top_face)
        self.last_out_of_reach.copy_(out_of_reach)

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

        hold_progress = (
            self.hold_counter.float() / max(float(self.cfg.hold_steps), 1.0)
        ).clamp(0.0, 1.0)

        log = self.extras.setdefault("log", {})
        log["DICE/alignment"] = alignment.mean()
        log["DICE/position_error"] = position_error.mean()
        log["DICE/angular_speed"] = angular_speed.mean()
        log["DICE/hold_progress"] = hold_progress.mean()
        log["DICE/success_rate_per_step"] = success.float().mean()
        log["DICE/commands_in_active_episode"] = self.successes.float().mean()
        log["DICE/drop_rate_per_step"] = out_of_reach.float().mean()

        if self.cfg.emit_step_metrics:
            # Cloned step-level tensors survive Isaac Lab's automatic reset and
            # are consumed only by frozen-policy evaluation and video scripts.
            self.extras["dice_success"] = self.last_success.clone()
            self.extras["dice_completed_face"] = self.last_completed_face.clone()
            self.extras["dice_success_latency_steps"] = self.last_success_latency.clone()
            self.extras["dice_target_face"] = self.target_faces.clone()
            self.extras["dice_top_face"] = self.last_top_face.clone()
            self.extras["dice_alignment"] = self.last_alignment.clone()
            self.extras["dice_position_error"] = self.last_position_error.clone()
            self.extras["dice_hold_progress"] = hold_progress.clone()
            self.extras["dice_commands_completed"] = self.successes.clone()
            self.extras["dice_drop"] = out_of_reach.clone()

        return reward

    # ------------------------------------------------------------------
    # Done and reset
    # ------------------------------------------------------------------

    def _get_dones(self):
        self._compute_intermediate_values()
        out_of_reach = torch.linalg.vector_norm(
            self.object_pos - self.in_hand_pos, dim=-1
        ) >= self.cfg.fall_dist
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.max_commands_per_episode > 0:
            command_limit = self.successes >= self.cfg.max_commands_per_episode
        else:
            command_limit = torch.zeros_like(out_of_reach)

        self.last_out_of_reach.copy_(out_of_reach)
        if self.cfg.emit_step_metrics:
            self.extras["dice_drop"] = out_of_reach.clone()
            self.extras["dice_commands_completed"] = self.successes.clone()
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
    # Commands
    # ------------------------------------------------------------------

    def _assign_targets(self, env_ids, advance):
        if env_ids.numel() == 0:
            return

        mode = self.cfg.target_mode.lower()
        current_targets = self.target_faces[env_ids]

        if mode == "fixed":
            targets = torch.full_like(current_targets, int(self.cfg.fixed_target_face))
        elif mode == "cycle":
            sequence = torch.tensor(
                self.cfg.target_sequence,
                dtype=torch.long,
                device=self.device,
            )
            if advance:
                self.sequence_index[env_ids] = (
                    self.sequence_index[env_ids] + 1
                ) % sequence.numel()
            else:
                self.sequence_index[env_ids] = 0
            targets = sequence[self.sequence_index[env_ids]]
        elif mode == "random":
            if advance:
                # Uniformly sample one of the other five faces. Replacing a
                # duplicate with the next face would double that transition's
                # probability and bias command sequencing.
                offsets = torch.randint(
                    1, 6, current_targets.shape, device=self.device
                )
                targets = (current_targets - 1 + offsets) % 6 + 1
            else:
                targets = torch.randint(
                    1, 7, current_targets.shape, device=self.device
                )
        else:
            raise ValueError("target_mode must be one of: fixed, random, cycle")

        self.target_faces[env_ids] = targets
        self.target_one_hot[env_ids] = functional.one_hot(
            targets - 1, num_classes=6
        ).float()
        self.goal_rot[env_ids] = canonical_goal_quaternion(targets, device=self.device)
        self.hold_counter[env_ids] = 0
        self.command_age_steps[env_ids] = 0

        goal_positions = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_positions, self.goal_rot)
        self.reset_goal_buf[env_ids] = False

    # ------------------------------------------------------------------
    # Runtime diagnostics
    # ------------------------------------------------------------------

    def get_task_metrics(self):
        return {
            "target_face": self.target_faces,
            "top_face": self.last_top_face,
            "alignment": target_face_alignment(self.object_rot, self.target_faces),
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
