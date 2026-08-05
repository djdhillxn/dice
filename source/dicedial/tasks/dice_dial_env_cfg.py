"""Environment configurations for DICE.

DICE trains one command-conditioned Shadow Hand policy on the final task from
iteration zero.  The environments differ only in their runtime purpose:

* ``DICE-Shadow-Train-v0``: stock instanceable DexCube, no domain randomization.
* ``DICE-Shadow-Eval-v0``: nominal frozen-policy evaluation, no randomization.
* ``DICE-Shadow-Robust-v0``: held-out object mass and friction randomization.
* ``DICE-Shadow-Play-v0``: one numbered die, deterministic command sequence,
  no randomization, and a presentation camera.
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
class DiceBaseEnvCfg(ShadowHandEnvCfg):
    """Shared final-task configuration.

    The inherited Shadow Hand task supplies the hand, stock instanceable cube,
    contacts, action controller, and reset logic. DICE retains the stock
    157-dimensional full observation and appends eight command features:

    * requested-face one-hot: 6
    * normalized hold progress: 1
    * requested-face alignment with world up: 1
    """

    observation_space = 165
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

    # Compact reward.  The completion bonus intentionally dominates dense
    # near-target reward so the policy prefers finishing and accepting a new
    # command over loitering just below the hold threshold.
    alignment_scale = 1.0
    alignment_power = 4.0
    position_error_scale = -5.0
    angular_speed_scale = -0.02
    angular_penalty_gate_deg = 30.0
    action_penalty_scale = -0.0002
    success_bonus = 250.0
    drop_penalty = -50.0

    # Per-environment tensors are unnecessary during PPO collection. Scalar
    # training logs remain active; detailed step metrics are enabled only for
    # frozen-policy evaluation and presentation.
    emit_step_metrics = False

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
    events = None
