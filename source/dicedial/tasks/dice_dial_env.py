"""Command-conditioned in-hand die reorientation for DICE."""

import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as functional

from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv

from dicedial.geometry import (
    FACE_NORMALS,
    canonical_goal_quaternion,
    rotate_vector_wxyz,
    rotation_6d_from_quaternion_wxyz,
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
        # In Isaac Lab 2.3.2, InHandManipulationEnv.__init__ constructs a
        # VisualizationMarkers USD PointInstancer unconditionally *after*
        # DirectRLEnv prints "Completed setting up the environment...". DICE
        # does not need that UI object for headless training/evaluation. Build
        # the same in-hand runtime state here so marker creation can be skipped
        # entirely when cfg.visualize_goal_marker is false.
        self._startup_log(cfg, "Entering DICE environment constructor.")
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        self._startup_log(cfg, "DirectRLEnv initialization complete; initializing Shadow Hand runtime buffers.")

        self.num_hand_dofs = self.hand.num_joints
        self.hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.prev_targets = torch.zeros_like(self.hand_dof_targets)
        self.cur_targets = torch.zeros_like(self.hand_dof_targets)

        self.actuated_dof_indices = sorted(
            self.hand.joint_names.index(joint_name) for joint_name in cfg.actuated_joint_names
        )
        self.prev_actions = torch.zeros(
            (self.num_envs, len(self.actuated_dof_indices)), dtype=torch.float, device=self.device
        )
        self.finger_bodies = sorted(
            self.hand.body_names.index(body_name) for body_name in cfg.fingertip_body_names
        )
        self.num_fingertips = len(self.finger_bodies)

        self._startup_log(cfg, "Reading Shadow Hand DOF limits from the PhysX tensor view...")
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]
        self._startup_log(cfg, "Shadow Hand DOF limits ready.")

        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.04
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)

        if cfg.visualize_goal_marker:
            self._startup_log(cfg, "Creating goal VisualizationMarkers (presentation mode).")
            self.goal_markers = VisualizationMarkers(cfg.goal_object_cfg)
            self._startup_log(cfg, "Goal VisualizationMarkers ready.")
        else:
            self.goal_markers = None
            self._startup_log(cfg, "Skipping goal VisualizationMarkers for headless training/evaluation.")

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self._startup_log(cfg, "Shadow Hand runtime buffers initialized.")

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
        self.previous_target_faces = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Match Isaac Lab's DirectRLEnv lifecycle: do not sample/reset task
        # state inside __init__. The RSL-RL wrapper calls env.reset() immediately
        # after gym.make() returns, and that first reset is where the stock
        # in-hand task initializes randomized goals and state-dependent buffers.
        #
        # Keep a valid deterministic placeholder command until then. This also
        # avoids touching object_rot before _compute_intermediate_values() has
        # populated the in-hand state tensors.
        self.target_faces.fill_(1)
        self.target_one_hot.zero_()
        self.target_one_hot[:, 0] = 1.0
        self._initial_reset_pending = True
        self._startup_log(cfg, "Initial semantic command deferred to first environment reset.")
        self._startup_log(cfg, "DICE environment constructor complete.")

    @staticmethod
    def _startup_log(cfg, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [DICE ENV] {message}"
        print(line, flush=True)
        log_dir = getattr(cfg, "log_dir", None)
        if log_dir:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            with (path / "startup.log").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def compute_full_observations(self):
        """Return a clean 121-dimensional frame-invariant actor observation.

        Features (121 total):
        - Normalized hand joint positions: 24
        - Scaled hand joint velocities: 24
        - Previous actions: 20
        - Fingertip positions relative to cube: 15
        - Fingertip linear velocities: 15
        - Cube position relative to in-hand center: 3
        - Cube linear velocity: 3
        - Cube angular velocity: 3
        - Continuous 6D cube rotation: 6
        - Commanded face normal in world frame: 3
        - Commanded face alignment with world up (+Z): 1
        - Rotation axis error vector (cross product with +Z): 3
        - Normalized hold progress: 1
        """

        normalized_joint_pos = (
            2.0
            * (self.hand_dof_pos - self.hand_dof_lower_limits)
            / (self.hand_dof_upper_limits - self.hand_dof_lower_limits)
            - 1.0
        )
        scaled_joint_vel = self.hand_dof_vel * 0.2

        relative_fingertip_pos = (
            self.fingertip_pos - self.object_pos.unsqueeze(1)
        ).reshape(self.num_envs, -1)
        fingertip_linvel = self.fingertip_linvel.reshape(self.num_envs, -1)

        relative_object_pos = self.object_pos - self.in_hand_pos
        cube_rotation_6d = rotation_6d_from_quaternion_wxyz(self.object_rot)

        face_normals_dev = FACE_NORMALS.to(self.device)
        commanded_local_normals = face_normals_dev[self.target_faces - 1]
        commanded_world_normals = rotate_vector_wxyz(self.object_rot, commanded_local_normals)

        alignment = commanded_world_normals[:, 2:3]

        world_up = torch.zeros_like(commanded_world_normals)
        world_up[:, 2] = 1.0
        axis_error = torch.cross(commanded_world_normals, world_up, dim=-1)

        hold_progress = (
            (self.hold_counter.float() / max(float(self.cfg.hold_steps), 1.0))
            .clamp(0.0, 1.0)
            .unsqueeze(-1)
        )

        return torch.cat(
            (
                normalized_joint_pos,
                scaled_joint_vel,
                self.actions,
                relative_fingertip_pos,
                fingertip_linvel,
                relative_object_pos,
                self.object_linvel,
                self.object_angvel,
                cube_rotation_6d,
                commanded_world_normals,
                alignment,
                axis_error,
                hold_progress,
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
        angular_speed_excess = torch.relu(angular_speed - self.cfg.success_angular_speed)
        angular_penalty = (
            self.cfg.angular_speed_scale
            * angular_speed_excess.square()
            * near_target.float()
        )

        action_penalty = self.cfg.action_penalty_scale * torch.sum(self.actions.square(), dim=-1)
        action_rate = self.actions - self.prev_actions
        action_rate_penalty_scale = getattr(self.cfg, "action_rate_penalty_scale", -0.01)
        action_rate_penalty = action_rate_penalty_scale * torch.sum(action_rate.square(), dim=-1)
        drop_penalty = self.cfg.drop_penalty * out_of_reach.float()

        target_rate = self.cur_targets - self.prev_targets
        target_rate_rms = torch.sqrt(target_rate.square().mean())
        action_rate_rms = torch.sqrt(action_rate.square().mean())

        joint_range = self.hand_dof_upper_limits - self.hand_dof_lower_limits
        near_lower = self.hand_dof_pos <= (self.hand_dof_lower_limits + 0.02 * joint_range)
        near_upper = self.hand_dof_pos >= (self.hand_dof_upper_limits - 0.02 * joint_range)
        joint_limit_saturation = (near_lower | near_upper).float().mean()

        success_angle = math.cos(math.radians(self.cfg.success_angle_deg))
        hold_condition = (
            (alignment >= success_angle)
            & (position_error <= self.cfg.success_position_tolerance)
            & (angular_speed <= self.cfg.success_angular_speed)
        )
        previous_hold_counter = self.hold_counter.clone()
        self.hold_counter = torch.where(
            hold_condition,
            self.hold_counter + 1,
            torch.zeros_like(self.hold_counter),
        )

        hold_progress = (
            self.hold_counter.float() / max(float(self.cfg.hold_steps), 1.0)
        ).clamp(0.0, 1.0)

        # Signed hold-progress shaping. Each valid step is rewarded, but losing
        # a partial hold pays back the accumulated shaping, preventing a policy
        # from farming repeated 1..19-step attempts without completing.
        hold_delta = (self.hold_counter - previous_hold_counter).float()
        hold_shaping = (
            self.cfg.hold_progress_scale
            * hold_delta
            / max(float(self.cfg.hold_steps), 1.0)
        )

        success = self.hold_counter >= self.cfg.hold_steps
        success_reward = self.cfg.success_bonus * success.float()

        reward = (
            progress_reward
            + hold_shaping
            + position_penalty
            + angular_penalty
            + action_penalty
            + action_rate_penalty
            + success_reward
            + drop_penalty
        )

        self.last_alignment.copy_(alignment)
        self.last_position_error.copy_(position_error)
        if current_top_face is not None:
            self.last_top_face.copy_(current_top_face)
        self.last_out_of_reach.copy_(out_of_reach)
        self.prev_actions.copy_(self.actions)

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

        self.max_hold_counter = torch.maximum(self.max_hold_counter, self.hold_counter)

        dt = getattr(self, "step_dt", 0.016666666)
        sim_time_min = ((self.episode_length_buf.float() + 1.0) * dt) / 60.0
        latencies_sec = self.last_success_latency * dt

        has_prev = self.previous_target_faces > 0
        is_opp = has_prev & ((self.previous_target_faces + self.target_faces) == 7)
        is_adj = has_prev & ((self.previous_target_faces + self.target_faces) != 7)

        opp_success = self.last_success[is_opp].float().mean() if is_opp.any() else torch.tensor(0.0, device=self.device)
        adj_success = self.last_success[is_adj].float().mean() if is_adj.any() else torch.tensor(0.0, device=self.device)
        failed_max_hold = self.max_hold_counter[~self.last_success].float().mean() if (~self.last_success).any() else torch.tensor(0.0, device=self.device)

        # Construct a FRESH dictionary on every step to fix RSL-RL logging aliasing!
        log = {}
        log["DICE/alignment"] = alignment.mean().item()
        log["DICE/position_error"] = position_error.mean().item()
        log["DICE/angular_speed"] = angular_speed.mean().item()
        log["DICE/hold_progress"] = hold_progress.mean().item()
        log["DICE/success_rate_per_step"] = success.float().mean().item()
        log["DICE/episode_has_at_least_1_success_rate"] = (self.successes >= 1).float().mean().item()
        log["DICE/commands_completed_per_episode"] = self.successes.float().mean().item()
        log["DICE/commands_per_sim_minute"] = (self.successes.float() / sim_time_min.clamp_min(0.01)).mean().item()
        log["DICE/success_within_2s_rate"] = (self.last_success & (latencies_sec <= 2.0)).float().mean().item()
        log["DICE/success_within_4s_rate"] = (self.last_success & (latencies_sec <= 4.0)).float().mean().item()
        log["DICE/success_within_6s_rate"] = (self.last_success & (latencies_sec <= 6.0)).float().mean().item()
        log["DICE/success_rate_adjacent_transitions"] = adj_success.item()
        log["DICE/success_rate_opposite_transitions"] = opp_success.item()
        log["DICE/max_hold_reached_failed_commands"] = failed_max_hold.item()
        log["DICE/drop_rate_per_step"] = out_of_reach.float().mean().item()
        log["DICE/drop_rate_per_sim_minute"] = (out_of_reach.float() / (dt / 60.0)).mean().item()
        log["DICE/gate_angle_rate"] = (alignment >= success_angle).float().mean().item()
        log["DICE/gate_position_rate"] = (position_error <= self.cfg.success_position_tolerance).float().mean().item()
        log["DICE/gate_angular_speed_rate"] = (angular_speed <= self.cfg.success_angular_speed).float().mean().item()
        log["DICE/gate_all_rate"] = hold_condition.float().mean().item()
        log["DICE/hold_ge_1_rate"] = (self.hold_counter >= 1).float().mean().item()
        log["DICE/hold_ge_5_rate"] = (self.hold_counter >= 5).float().mean().item()
        log["DICE/hold_ge_10_rate"] = (self.hold_counter >= 10).float().mean().item()
        log["DICE/hold_ge_19_rate"] = (self.hold_counter >= 19).float().mean().item()
        log["DICE/reward_alignment_progress"] = progress_reward.mean().item()
        log["DICE/reward_hold_shaping"] = hold_shaping.mean().item()
        log["DICE/reward_position"] = position_penalty.mean().item()
        log["DICE/reward_angular"] = angular_penalty.mean().item()
        log["DICE/reward_action"] = action_penalty.mean().item()
        log["DICE/reward_action_rate"] = action_rate_penalty.mean().item()
        log["DICE/reward_success"] = success_reward.mean().item()
        log["DICE/reward_drop"] = drop_penalty.mean().item()
        log["DICE/reward_total"] = reward.mean().item()
        log["DICE/action_mean_abs"] = self.actions.abs().mean().item()
        log["DICE/action_rms"] = torch.sqrt(self.actions.square().mean()).item()
        log["DICE/action_rate_rms"] = action_rate_rms.item()
        log["DICE/target_rate_rms"] = target_rate_rms.item()
        log["DICE/joint_limit_saturation"] = joint_limit_saturation.item()
        log["DICE/action_saturation_rate"] = (self.actions.abs() >= 0.999).float().mean().item()

        self.extras["log"] = log

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
        initial_reset = getattr(self, "_initial_reset_pending", False)
        if initial_reset:
            self._startup_log(self.cfg, "First reset: entering Shadow Hand reset.")

        super()._reset_idx(env_ids)

        if initial_reset:
            self._startup_log(self.cfg, "First reset: parent Shadow Hand reset complete.")

        self.hold_counter[env_ids] = 0
        self.command_age_steps[env_ids] = 0
        self.prev_actions[env_ids] = 0.0
        self.last_success_latency[env_ids] = 0.0
        self.last_success[env_ids] = False
        self.last_out_of_reach[env_ids] = False
        self.last_completed_face[env_ids] = 0
        self.sequence_index[env_ids] = 0
        self.previous_target_faces[env_ids] = 0
        self.max_hold_counter[env_ids] = 0
        self.last_alignment[env_ids] = target_face_alignment(
            self.object_rot[env_ids], self.target_faces[env_ids]
        )
        if initial_reset:
            self._startup_log(self.cfg, "First reset: DICE reset bookkeeping complete.")

    def _reset_target_pose(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        initial_reset = getattr(self, "_initial_reset_pending", False)
        if initial_reset:
            self._startup_log(self.cfg, "First reset: sampling initial semantic face commands.")
        self._assign_targets(env_ids, advance=False)
        if initial_reset:
            self._initial_reset_pending = False
            self._startup_log(self.cfg, "First reset: semantic face commands ready.")

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
                if getattr(self, "_initial_reset_pending", False):
                    self._startup_log(self.cfg, f"First reset: torch.randint target sampling on {self.device}.")
                targets = torch.randint(
                    1, 7, current_targets.shape, device=self.device
                )
                if getattr(self, "_initial_reset_pending", False):
                    self._startup_log(self.cfg, "First reset: target IDs sampled.")
        else:
            raise ValueError("target_mode must be one of: fixed, random, cycle")

        initial_reset = getattr(self, "_initial_reset_pending", False)
        self.target_faces[env_ids] = targets
        self.target_one_hot[env_ids] = functional.one_hot(
            targets - 1, num_classes=6
        ).float()
        if initial_reset:
            self._startup_log(self.cfg, "First reset: one-hot target encoding ready.")
        self.goal_rot[env_ids] = canonical_goal_quaternion(targets, device=self.device)
        if initial_reset:
            self._startup_log(self.cfg, "First reset: canonical goal quaternions ready.")
        self.hold_counter[env_ids] = 0
        self.command_age_steps[env_ids] = 0
        if advance:
            self.previous_target_faces[env_ids] = current_targets
        else:
            self.previous_target_faces[env_ids] = 0
        self.max_hold_counter[env_ids] = 0

        if self.cfg.visualize_goal_marker:
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
