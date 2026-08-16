"""Environment configurations for DICE.

DICE trains one command-conditioned Shadow Hand policy on the final task from
iteration zero.  The environments differ only in their runtime purpose:

* ``DICE-Shadow-Train-v0``: stock instanceable DexCube, no domain randomization.
* ``DICE-Shadow-Eval-v0``: nominal frozen-policy evaluation, no randomization.
* ``DICE-Shadow-Robust-v0``: held-out object mass and friction randomization.
* ``DICE-Shadow-Adverse-v0``: fixed heavy, low-friction material stress test.
* ``DICE-Shadow-Play-v0``: one numbered die, deterministic command sequence,
  no randomization, and a presentation camera.
* ``DICE-Shadow-Play-Robust-v0``: numbered-die playback with the held-out
  symmetric material distribution.
* ``DICE-Shadow-Play-Adverse-v0``: numbered-die playback at the fixed adverse
  heavy, low-friction corner.
"""

from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_NUMBERED_DIE_USD = str(_PACKAGE_ROOT / "assets" / "numbered_die.usda")


def _numbered_die_spawn():
    """Create a video-only die with stock-cube-aligned physical properties."""

    return sim_utils.UsdFileCfg(
        usd_path=_NUMBERED_DIE_USD,
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
        mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        semantic_tags=[("class", "die")],
    )


@configclass
class DiceRobustObjectEventsCfg:
    """Held-out object randomization used only by the robustness evaluator."""

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=1,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
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
class DiceNominalObjectEventsCfg:
    """Explicit nominal material contract for the presentation die."""

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=1,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )


@configclass
class DiceAdverseObjectEventsCfg:
    """Fixed heavy/slippery object properties for the final stress test.

    This is deliberately an adverse corner rather than another symmetric
    distribution: the object is 1.5x its nominal mass and both surface-friction
    coefficients are fixed at 0.7. Keeping the condition fixed makes failures
    directly attributable to the material shift instead of to a lucky mixture
    of easier and harder samples.
    """

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=1,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 0.7),
            "dynamic_friction_range": (0.7, 0.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )

    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        min_step_count_between_reset=1,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class DiceBaseEnvCfg(ShadowHandEnvCfg):
    """Shared final-task configuration.

    The inherited Shadow Hand task supplies the hand, stock instanceable cube,
    contacts, action controller, and reset logic. DICE uses a rebuilt,
    task-aligned 126-dimensional actor observation and an asymmetric critic.
    """

    observation_space = 126
    action_space = 20
    state_space = 247
    asymmetric_obs = True
    obs_type = "full"

    scene = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    episode_length_s = 24.0

    target_mode = "random"  # random | cycle | fixed
    fixed_target_face = 1
    target_sequence = (1, 6, 3, 5, 2, 4)
    switch_target_on_success = True
    max_commands_per_episode = 0

    # Final task definition used from the first training transition.
    success_angle_deg = 16.0
    success_position_tolerance = 0.12
    success_angular_speed = 1.25
    hold_steps = 20

    # Rotation-progress reward shaping: multiplies angular-error reduction (radians).
    # Moving from 90 deg error to 16 deg yields ~52 raw reward points.
    rotation_progress_scale = 40.0
    alignment_scale = 40.0
    # Reward signed changes in the consecutive hold counter. A valid hold step
    # earns +2.0, while breaking a k-step partial hold claws back that progress.
    hold_progress_scale = 40.0
    position_error_scale = -2.0
    # Smooth settling penalty on squared angular speed near the target face (45 deg to 16 deg).
    angular_speed_scale = -0.05
    angular_penalty_gate_deg = 45.0
    action_penalty_scale = 0.0
    action_rate_penalty_scale = -0.01
    action_bound_penalty_scale = -0.1
    success_bonus = 250.0
    drop_penalty = -100.0
    reward_global_scale = 0.1

    # Converts the fingertip reaction-force norm into a bounded [0, 1] actor
    # load proxy. The training log records its mean and saturation fraction so
    # this scale can be calibrated from real simulator values.
    fingertip_load_scale = 0.1

    # Per-environment tensors are unnecessary during PPO collection. Scalar
    # training logs remain active; detailed step metrics are enabled only for
    # frozen-policy evaluation and presentation.
    emit_step_metrics = False

    # Goal markers are presentation-only. Updating thousands of marker poses is
    # unnecessary in headless training/evaluation and can dominate startup/reset.
    visualize_goal_marker = False

    # DirectRLEnv normally resets terminal environments inside ``step`` before
    # an outer video wrapper requests its frame. Presentation environments can
    # opt out for the final terminal transition so a six-command hold or an
    # adverse drop remains visible. Training and evaluation always keep the
    # default automatic-reset behavior.
    defer_terminal_reset_for_capture = False

    # Low-pass joint targets. The inherited full-observation Shadow Hand uses
    # 1.0, but precise settling benefits from the 0.3 smoothing used by Isaac
    # Lab's OpenAI-style Shadow Hand configuration.
    act_moving_average = 0.3

    # Nominal training/evaluation use the inherited stock DexCube and no
    # randomization.  Robust evaluation enables its own event configuration.
    events = None


@configclass
class DiceTrainEnvCfg(DiceBaseEnvCfg):
    """Primary training environment: the complete final task, unchanged."""

    target_mode = "random"
    switch_target_on_success = True


@configclass
class DiceEvalEnvCfg(DiceBaseEnvCfg):
    """Nominal frozen-policy evaluation environment."""

    scene = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    target_mode = "random"
    switch_target_on_success = True
    emit_step_metrics = True


@configclass
class DiceRobustEnvCfg(DiceEvalEnvCfg):
    """Held-out evaluation with ±20% object mass/friction variation."""

    events = DiceRobustObjectEventsCfg()


@configclass
class DiceAdverseEnvCfg(DiceEvalEnvCfg):
    """Held-out evaluation at a fixed heavy, low-friction adverse corner."""

    events = DiceAdverseObjectEventsCfg()


@configclass
class DicePlayEnvCfg(DiceBaseEnvCfg):
    """Single numbered-die environment for deterministic video rendering."""

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

    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=_numbered_die_spawn(),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.39, 0.6),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target_mode = "cycle"
    target_sequence = (1, 6, 3, 5, 2, 4)
    switch_target_on_success = True
    max_commands_per_episode = 6
    episode_length_s = 40.0
    emit_step_metrics = True
    visualize_goal_marker = False
    defer_terminal_reset_for_capture = True
    events = DiceNominalObjectEventsCfg()


@configclass
class DicePlayRobustEnvCfg(DicePlayEnvCfg):
    """Numbered-die presentation under symmetric held-out physics variation."""

    events = DiceRobustObjectEventsCfg()


@configclass
class DicePlayAdverseEnvCfg(DicePlayEnvCfg):
    """Numbered-die presentation at the fixed heavy/slippery corner."""

    # Continue the command cycle until the object drops or the same 24-second
    # horizon used by final evaluation expires. This exposes the observed
    # long-horizon retention boundary instead of stopping after six commands.
    max_commands_per_episode = 0
    episode_length_s = 24.0
    events = DiceAdverseObjectEventsCfg()
