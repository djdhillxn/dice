"""Gymnasium registrations for DICE."""

import gymnasium as gym


_REGISTRATIONS = {
    "DICE-Shadow-Train-v0": "dicedial.tasks.dice_dial_env_cfg:DiceTrainEnvCfg",
    "DICE-Shadow-Eval-v0": "dicedial.tasks.dice_dial_env_cfg:DiceEvalEnvCfg",
    "DICE-Shadow-Robust-v0": "dicedial.tasks.dice_dial_env_cfg:DiceRobustEnvCfg",
    "DICE-Shadow-Play-v0": "dicedial.tasks.dice_dial_env_cfg:DicePlayEnvCfg",
}

for environment_id, config_entry_point in _REGISTRATIONS.items():
    if environment_id not in gym.registry:
        gym.register(
            id=environment_id,
            entry_point="dicedial.tasks.dice_dial_env:DiceEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": config_entry_point,
                "rsl_rl_cfg_entry_point": (
                    "dicedial.agents.rsl_rl_ppo_cfg:DicePPORunnerCfg"
                ),
            },
        )
