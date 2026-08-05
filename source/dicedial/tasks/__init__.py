"""Gymnasium registrations for DiceDial."""

import gymnasium as gym


_REGISTRATIONS = {
    "DiceDial-Shadow-Easy-v0": "dicedial.tasks.dice_dial_env_cfg:DiceDialEasyEnvCfg",
    "DiceDial-Shadow-Random-v0": "dicedial.tasks.dice_dial_env_cfg:DiceDialRandomEnvCfg",
    "DiceDial-Shadow-Sequence-v0": "dicedial.tasks.dice_dial_env_cfg:DiceDialSequenceEnvCfg",
    "DiceDial-Shadow-Robust-v0": "dicedial.tasks.dice_dial_env_cfg:DiceDialRobustEnvCfg",
    "DiceDial-Shadow-Play-v0": "dicedial.tasks.dice_dial_env_cfg:DiceDialPlayEnvCfg",
}

for environment_id, config_entry_point in _REGISTRATIONS.items():
    if environment_id not in gym.registry:
        gym.register(
            id=environment_id,
            entry_point="dicedial.tasks.dice_dial_env:DiceDialEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": config_entry_point,
                "sb3_cfg_entry_point": "dicedial.agents:sb3_ppo_cfg.yaml",
            },
        )
