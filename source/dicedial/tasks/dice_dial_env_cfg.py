"""Configuration variants for the command-conditioned die task."""

import os
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DIE_USD = str(_PACKAGE_ROOT / "assets" / "numbered_die.usda")


def _numbered_die_spawn(scale=(1.0, 1.0, 1.0)):
    return sim_utils.UsdFileCfg(
        usd_path=_DIE_USD,
        scale=scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            max_depenetration_velocity=1000.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
    )


class DiceDialObjectRandomizationCfg:
    """Mild object-only randomization for held-out robustness evaluation."""

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=1,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        min_step_count_between_reset=1,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class DiceDialBaseEnvCfg(ShadowHandEnvCfg):
    """Base DiceDial configuration.

    The inherited task supplies the Shadow Hand, action controller, fingertip
    sensing, object resets, scene cloning, and direct vectorized environment.
    DiceDial only changes the goal semantics and reward.
    """

    # Base Shadow Hand full observation (157) + target one-hot (6) + hold (1).
    observation_space = 164
    action_space = 20
    state_space = 0
    obs_type = "full"

    scene = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    episode_length_s = 20.0

    # Command behavior.
    target_mode = "random"  # fixed, random, or cycle
    fixed_target_face = 1
    target_sequence = (1, 6, 3, 5, 2, 4)
    switch_target_on_success = True
    max_commands_per_episode = 0  # zero disables this terminal condition

    # Hold-to-confirm success.
    success_angle_deg = 16.0
    success_position_tolerance = 0.12
    success_angular_speed = 1.25
    hold_steps = 20

    # Reward terms.
    alignment_scale = 4.0
    alignment_power = 4.0
    position_error_scale = -2.0
    angular_speed_scale = -0.08
    action_penalty_scale = -0.002
    hold_bonus_scale = 0.25
    success_bonus = 12.0
    wrong_face_penalty_scale = -0.15

    # A local numbered die is the default. Set DICEDIAL_USE_STOCK_CUBE=1 to
    # troubleshoot asset loading while preserving the task logic.
    if os.getenv("DICEDIAL_USE_STOCK_CUBE", "0") == "1":
        object_cfg = ShadowHandEnvCfg.object_cfg
        goal_object_cfg = ShadowHandEnvCfg.goal_object_cfg
    else:
        object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/object",
            spawn=_numbered_die_spawn(),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, -0.39, 0.6),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        goal_object_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/goal_marker",
            markers={
                "goal": sim_utils.UsdFileCfg(
                    usd_path=_DIE_USD,
                    scale=(1.0, 1.0, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.15, 0.8, 0.35),
                        opacity=0.45,
                    ),
                )
            },
        )


@configclass
class DiceDialEasyEnvCfg(DiceDialBaseEnvCfg):
    """Stage 1: hold one fixed face with a relaxed success gate."""

    target_mode = "fixed"
    fixed_target_face = 1
    switch_target_on_success = False
    episode_length_s = 12.0
    success_angle_deg = 20.0
    success_angular_speed = 1.8
    hold_steps = 14
    success_bonus = 8.0


@configclass
class DiceDialRandomEnvCfg(DiceDialBaseEnvCfg):
    """Stage 2: all six commanded faces with randomized resets."""

    target_mode = "random"
    switch_target_on_success = True
    episode_length_s = 16.0


@configclass
class DiceDialSequenceEnvCfg(DiceDialBaseEnvCfg):
    """Stage 3: successive commands without releasing the die."""

    target_mode = "random"
    switch_target_on_success = True
    episode_length_s = 24.0


@configclass
class DiceDialRobustEnvCfg(DiceDialSequenceEnvCfg):
    """Evaluation task with mild held-out die mass and friction variation."""

    events: DiceDialObjectRandomizationCfg = DiceDialObjectRandomizationCfg()


@configclass
class DiceDialPlayEnvCfg(DiceDialBaseEnvCfg):
    """Single-environment deterministic command sequence for rendering."""

    viewer = ViewerCfg(
        eye=(0.95, -1.20, 0.88),
        lookat=(0.0, -0.39, 0.59),
        origin_type="world",
        resolution=(1280, 720),
    )
    scene = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    target_mode = "cycle"
    target_sequence = (1, 6, 3, 5, 2, 4)
    switch_target_on_success = True
    max_commands_per_episode = 0
    episode_length_s = 40.0
