"""Small SB3 callbacks specific to DiceDial."""

from pathlib import Path

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class DiceDialMetricsCallback(BaseCallback):
    """Write task diagnostics to TensorBoard and a compact CSV trace."""

    def __init__(self, raw_env, output_dir, sample_every=1000, verbose=0):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.output_dir = Path(output_dir)
        self.sample_every = max(int(sample_every), 1)
        self.rows = []

    def _on_step(self):
        if self.n_calls % self.sample_every != 0:
            return True

        metrics = self.raw_env.unwrapped.get_task_metrics()
        row = {"timesteps": int(self.num_timesteps)}
        for key, value in metrics.items():
            scalar = value.float().mean().item()
            row[key] = scalar
            self.logger.record("dicedial/" + key, scalar)

        self.rows.append(row)
        if len(self.rows) % 10 == 0:
            self._flush()
        return True

    def _on_training_end(self):
        self._flush()

    def _flush(self):
        if not self.rows:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.output_dir / "task_metrics.csv", index=False)
