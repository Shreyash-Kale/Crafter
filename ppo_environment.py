# ppo_environment.py – Gymnasium adapter for Crafter + per-episode CSV dump

import os, numpy as np, pandas as pd
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import crafter
import gymnasium as gym
from gymnasium import spaces
import glob


# ────────────────────────────────────────────────────────────────────────
# Extra recorder that also saves episode.csv
# ────────────────────────────────────────────────────────────────────────
class CSVRecorder(crafter.Recorder):
    """
    Extends crafter.Recorder so _save() also writes episode.csv
    containing one row per environment step.
    """

    def _save(self):
        super()._save()                       # keep npz / mp4 / stats.json

        steps = len(self._actions)
        if steps == 0:        # safety guard
            return

        results = {
            "time_step": np.arange(steps),
            "action": self._actions,
            "reward": self._rewards,
            "cumulative_reward": np.cumsum(self._rewards),
        }

        # inventory columns
        inv_series = [info.get("inventory", {}) for info in self._infos]
        for k in {k for d in inv_series for k in d}:
            results[f"inv_{k}"] = [d.get(k, 0) for d in inv_series]

        # achievement columns (bool → 0/1)
        ach_series = [info.get("achievements", {}) for info in self._infos]
        for k in {k for d in ach_series for k in d}:
            results[f"ach_{k}"] = [int(d.get(k, False)) for d in ach_series]

        latest_npz = max(glob.glob(os.path.join(self._dir, "*.npz")),
                 key=os.path.getmtime)
        base = os.path.splitext(os.path.basename(latest_npz))[0]        # 20250425T194006-ach5-len147
        pd.DataFrame(results).to_csv(os.path.join(self._dir, f"{base}.csv"),
                                    index=False)




# ────────────────────────────────────────────────────────────────────────
class RecorderAdapter(gym.Env):
    """Wrap CSVRecorder so Stable-Baselines3 (Gymnasium) accepts it."""

    metadata = {}

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

        # ACTION space
        act = recorder.action_space
        self.action_space = spaces.Discrete(act.n) if hasattr(act, "n") else act

        # OBSERVATION space
        obs = recorder.observation_space
        self.observation_space = (
            spaces.Box(low=0, high=255, shape=obs.shape, dtype=np.uint8)
            if not isinstance(obs, spaces.Box) else obs
        )

    # Gymnasium API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            try:
                self.recorder.env.seed(seed)
            except AttributeError:
                pass
        return self.recorder.reset(), {}

    def step(self, action):
        obs, rew, done, info = self.recorder.step(action)
        return obs, rew, bool(done), False, info

    def render(self, *a, **k):
        return self.recorder.render(*a, **k)

    def close(self):
        self.recorder.close()


# ────────────────────────────────────────────────────────────────────────
def _ffmpeg_available() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401
        return True
    except ImportError:
        return False


def create_environment(output_dir: str = "./results", save_video: bool = True):
    """
    Return a Crafter env wrapped for SB3, saving episode.csv per episode.
    """
    if save_video and not _ffmpeg_available():
        print("⚠️  imageio-ffmpeg not found → disabling video recording.")
        save_video = False

    env = crafter.Env()
    env = CSVRecorder(
        env,
        output_dir,
        save_stats=True,
        save_video=save_video,
        save_episode=True,
    )
    return RecorderAdapter(env)
