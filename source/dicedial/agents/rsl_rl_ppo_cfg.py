"""RSL-RL ``OnPolicyRunner`` configuration for DiceDial.

This dict is consumed directly by ``rsl_rl.runners.OnPolicyRunner`` via the
training script ``scripts/train_rsl.py``.

Key design decisions
--------------------
* ``num_steps_per_env = 128``  (SB3 used 16)
  Longer rollouts dramatically improve credit assignment for the delayed
  success bonus (+12) which may be hundreds of steps away.  With 2 048 envs
  this yields 262 144 transitions per update — still fast on GPU.

* ``num_mini_batches = 4`` → mini-batch size = 2048 × 128 / 4 = 65 536.
  Large enough for stable gradient estimates with the 171-dim observation.

* ``schedule = "adaptive"``
  RSL-RL adjusts the learning rate to keep KL divergence near ``desired_kl``.
  This removes the need to hand-tune an LR schedule across curriculum levels.

* Architecture ``[512, 512, 256, 128]``
  Matches the SB3 config so warm-start from an existing SB3 checkpoint
  (manual weight copy) is feasible if needed.
"""

DICEDIAL_RSL_RL_CFG: dict = {
    "runner": {
        "policy_class_name": "ActorCritic",
        "algorithm_class_name": "PPO",
        # Steps collected per environment per update.
        "num_steps_per_env": 128,
        # Total iterations (each is one full rollout + update).
        # Override via --max_iterations in train_rsl.py.
        "max_iterations": 50_000,
        # Save a checkpoint every N iterations.
        "save_interval": 500,
        # Run name tag appended to the log directory.
        "run_name": "",
        "experiment_name": "DiceDial",
        "resume": False,
        "load_run": -1,
        "checkpoint": -1,
    },
    "policy": {
        "init_noise_std": 1.0,
        # Actor and critic share the same hidden architecture.
        "actor_hidden_dims": [512, 512, 256, 128],
        "critic_hidden_dims": [512, 512, 256, 128],
        "activation": "elu",
    },
    "algorithm": {
        # PPO clip coefficient.
        "clip_param": 0.2,
        # Entropy bonus — encourages exploration over all 6 faces.
        "entropy_coef": 0.005,
        # Critic loss weight.
        "value_loss_coef": 1.0,
        # Use a clipped value loss for training stability.
        "use_clipped_value_loss": True,
        # Gradient epochs per rollout.
        "num_learning_epochs": 5,
        # Mini-batches per epoch.
        "num_mini_batches": 4,
        # Base learning rate (adaptive schedule will adjust this).
        "learning_rate": 3.0e-4,
        # "adaptive" adjusts LR to keep KL near desired_kl.
        "schedule": "adaptive",
        # Discount factor.  γ=0.99 over 128 steps discounts the success bonus
        # to ~0.99^128 ≈ 27 % — far better than γ=0.99^16 ≈ 85 % in SB3.
        "gamma": 0.99,
        # GAE lambda.
        "lam": 0.95,
        # Target KL for the adaptive LR schedule.
        "desired_kl": 0.02,
        # Maximum gradient norm.
        "max_grad_norm": 1.0,
    },
}
