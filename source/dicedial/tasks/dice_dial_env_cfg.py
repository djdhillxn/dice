"""Configuration variants for the command-conditioned die task.

Training uses ``DiceDial-Shadow-Sequence-v0`` with a single continuous run.
The Automatic Curriculum Learning (ACL) manager in ``dicedial.curriculum``
morphs the success thresholds in place as the policy improves — no separate
stage environments are required.  ``DiceDial-Shadow-Robust-v0`` is kept as a
held-out evaluation variant.  ``DiceDial-Shadow-Play-v0`` is for video
rendering only.
"""

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
    """Die mass and friction randomization applied at every reset.

    Applied in the base training config so the policy is exposed to physical
    variation from step one.  The Robust eval variant keeps the same events but
    can be evaluated independently to separate robustness from competence.
    """

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

    Domain randomization (mass ±20 %, friction ±20 %) is active from the very
    first reset.  The ACL manager will tighten ``success_angle_deg``,
    ``hold_steps``, ``success_angular_speed``, and ``angular_speed_scale``
    progressively during training.
    """

    # Base Shadow Hand full observation (157) + target one-hot (6) + hold (1)
    # + commanded face normal in world (3) + top face normal in world (3)
    # + alignment scalar (1) = 171.
    observation_space = 171
    action_space = 20
    state_space = 0
    obs_type = "full"

    scene = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    episode_length_s = 24.0

    # Command behaviour.
    target_mode = "random"          # fixed | random | cycle
    fixed_target_face = 1
    target_sequence = (1, 6, 3, 5, 2, 4)
    switch_target_on_success = True
    max_commands_per_episode = 0    # zero disables this terminal condition

    # --- Hold-to-confirm thresholds (ACL Level 0 — relaxed) ---
    # The AclCurriculum callback will overwrite these mid-training.
    # Final values (Level 3): success_angle_deg=16, hold_steps=20,
    # success_angular_speed=1.25.
    success_angle_deg = 30.0
    success_position_tolerance = 0.12
    success_angular_speed = 2.0
    hold_steps = 8

    # --- Reward terms ---
    alignment_scale = 4.0
    alignment_power = 4.0
    position_error_scale = -2.0
    # angular_speed_scale starts mild; ACL tightens it alongside the gate.
    angular_speed_scale = -0.02
    action_penalty_scale = -0.002
    hold_bonus_scale = 0.25
    success_bonus = 12.0
    # Wrong-face penalty is now *stable-gated* in the env — it only fires
    # when angular speed is below success_angular_speed, so the policy is
    # never penalised for transitioning through other faces during rotation.
    wrong_face_penalty_scale = -0.15

    # Domain randomization — active from day one.
    events: DiceDialObjectRandomizationCfg = DiceDialObjectRandomizationCfg()

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


# ---------------------------------------------------------------------------
# Training environment — the only config used for live training.
# ACL starts it at Level 0 (relaxed thresholds) and ratchets automatically.
# ---------------------------------------------------------------------------

@configclass
class DiceDialSequenceEnvCfg(DiceDialBaseEnvCfg):
    """Main training task: successive commands without releasing the die.

    Thresholds start relaxed (ACL Level 0) and are tightened automatically by
    the AclCurriculum callback as the policy's commands-per-episode mean rises.
    """

    target_mode = "random"
    switch_target_on_success = True
    episode_length_s = 24.0


# ---------------------------------------------------------------------------
# Evaluation-only variants
# ---------------------------------------------------------------------------

@configclass
class DiceDialRobustEnvCfg(DiceDialSequenceEnvCfg):
    """Evaluation task with stronger held-out die mass and friction variation.

    Uses the same DR events as the base config but documents intent: this env
    is never used for training and always evaluated with a frozen checkpoint.
    Nominal and robustness results must be reported separately.
    """

    # No extra changes — DiceDialBaseEnvCfg already has DR events at ±20 %.
    # For a stricter held-out test, consider widening to ±30 % here.
    pass


@configclass
class DiceDialPlayEnvCfg(DiceDialBaseEnvCfg):
    """Single-environment deterministic command sequence for video rendering."""

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
    # Play uses the final (Level 3) thresholds for a fair visual demo.
    success_angle_deg = 16.0
    hold_steps = 20
    success_angular_speed = 1.25
    angular_speed_scale = -0.08
