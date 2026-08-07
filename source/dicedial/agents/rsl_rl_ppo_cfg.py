"""Official Shadow Hand PPO defaults for DICE."""

import importlib.metadata as metadata

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DicePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL configuration aligned with Isaac Lab's Shadow Hand baseline."""

    num_steps_per_env = 64
    max_iterations = 10_000
    save_interval = 250
    experiment_name = "DICE"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )


def installed_rsl_version():
    """Return the installed RSL-RL version without imposing a package pin."""

    for package_name in ("rsl-rl-lib", "rsl-rl"):
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return None


def make_runner_cfg(seed=42, device="cuda:0", max_iterations=None, run_name=""):
    """Build a runner config and apply Isaac Lab's compatibility conversion."""

    cfg = DicePPORunnerCfg()
    cfg.seed = int(seed)
    cfg.device = device
    cfg.run_name = run_name or ""
    if max_iterations is not None:
        cfg.max_iterations = int(max_iterations)

    version_string = installed_rsl_version()
    if version_string is not None:
        try:
            from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        except ImportError:
            pass
        else:
            cfg = handle_deprecated_rsl_rl_cfg(cfg, version_string)

    return cfg


def compatible_checkpoint_path(path):
    """Convert older checkpoints only when the installed Isaac Lab requires it."""

    version_string = installed_rsl_version()
    if version_string is None:
        return str(path)

    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_checkpoint
    except ImportError:
        return str(path)

    return handle_deprecated_rsl_rl_checkpoint(str(path), version_string)
