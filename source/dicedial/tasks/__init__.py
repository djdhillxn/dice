"""Gymnasium registrations for DiceDial.

Training uses ``DiceDial-Shadow-Sequence-v0``.  The ACL manager in
``dicedial.curriculum`` tightens thresholds automatically during a single run
— no separate Easy or Random stage environments are needed.
"""

import gymnasium as gym


_REGISTRATIONS = {
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
                # SB3 fallback (used by evaluate.py / play.py)
                "sb3_cfg_entry_point": "dicedial.agents:sb3_ppo_cfg.yaml",
                # RSL-RL primary trainer
                "rsl_rl_cfg_entry_point": "dicedial.agents.rsl_rl_ppo_cfg:DICEDIAL_RSL_RL_CFG",
            },
        )
