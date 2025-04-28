# train_ppo_crafter.py – episode_log.csv now includes decision-attribution columns

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from ppo_environment import create_environment


# ------------------------------------------------------------------ #
REF_COLUMNS = [
    # --- core step info ---
    "time_step", "action", "reward", "cumulative_reward",
    # --- decision attribution ---
    "action_prob", "entropy", "value", "advantage",
    # --- agent stats ---
    "health", "food", "drink", "energy",
    # --- inventory counts ---
    "sapling", "wood", "stone", "coal", "iron", "diamond",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
    # --- achievements (binary flags) ---
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood",
    "defeat_skeleton", "defeat_zombie",
    "eat_cow", "eat_plant",
    "make_iron_pickaxe", "make_iron_sword",
    "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword",
    "place_furnace", "place_plant", "place_stone", "place_table",
    "wake_up",
    # --- for completeness ---
    "step_reward",
]

all_possible_achievement_keys = [
    'make_sword', 'collect_coal', 'drink_water', 'eat_plant', 'collect_wood',
    'collect_stone', 'make_axe', 'defeat_zombie', 'make_pickaxe',
    'defeat_skeleton', 'collect_iron', 'defeat_cow', 'defeat_spider', 'make_armor'
]
# ------------------------------------------------------------------ #


class StepCSVCallback(BaseCallback):
    """
    Collect per-step data so episode_log.csv matches Crafter format
    and now also includes PPO decision-attribution metrics.
    """

    def __init__(self, csv_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.step_buffer = []
        self.episode_dfs = []

    def _on_step(self) -> bool:
        # only one env (DummyVecEnv n_envs=1)
        local = self.locals
        info = local["infos"][0]
        reward = float(local["rewards"][0])
        action = int(local["actions"][0])
        time_step = len(self.step_buffer)
        inventory = info.get("inventory", {})
        health = inventory.get("health", 0)


        # ---- decision-attribution pieces ---------------------------------
        log_prob = local["log_probs"][0].item()
        action_prob = float(np.exp(log_prob))
        entropy = -log_prob * action_prob
        value = local["values"][0].item()
        # Advantage is only filled after GAE; may be absent early
        advantage = float(local.get("advantages", [0])[0])

        row = {col: 0 for col in REF_COLUMNS}
        row.update(
            {
                "time_step": time_step,
                "action": action,
                "reward": reward,
                "step_reward": reward,
                "action_probability": action_prob,
                "entropy": entropy,
                "value": value,
                "advantage": advantage,
                "health": health
            }
        )
        if time_step == 0:
            row["cumulative_reward"] = reward
        else:
            row["cumulative_reward"] = self.step_buffer[-1][
                "cumulative_reward"
            ] + reward

        # ---- scalar player stats ----------------------------------------
        for key in ["health", "food", "drink", "energy"]:
            if key in info:
                row[key] = info[key]

        # ---- inventory tally --------------------------------------------
        inv = info.get("inventory", {})
        for res in [
            "sapling", "wood", "stone", "coal", "iron", "diamond",
            "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
            "wood_sword", "stone_sword", "iron_sword",
        ]:
            row[res] = inv.get(res, 0)

        # ---- achievements flags -----------------------------------------
        ach = info.get("achievements", {})
        for a in [
            "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
            "collect_sapling", "collect_stone", "collect_wood",
            "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
            "make_iron_pickaxe", "make_iron_sword",
            "make_stone_pickaxe", "make_stone_sword",
            "make_wood_pickaxe", "make_wood_sword",
            "place_furnace", "place_plant", "place_stone", "place_table",
            "wake_up",
        ]:
            row[a] = int(ach.get(a, False))

        self.step_buffer.append(row)

        # flush at end of episode
        if local["dones"][0]:
            # Add achievement info to the episode info dict for the Monitor wrapper
            achievements = {}
            for a in [
                "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
                "collect_sapling", "collect_stone", "collect_wood",
                "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
                "make_iron_pickaxe", "make_iron_sword",
                "make_stone_pickaxe", "make_stone_sword",
                "make_wood_pickaxe", "make_wood_sword",
                "place_furnace", "place_plant", "place_stone", "place_table",
                "wake_up",
            ]:
                achievements[a] = row[a]
            
            # Add to info dict for Monitor to capture
            local["infos"][0]["achievements"] = achievements
            
            # Add reward components
            local["infos"][0]["component_reward"] = row["reward"]
            local["infos"][0]["component_discount"] = 1.0  # Default value
            
            
            # Original code continues...
            df = pd.DataFrame(self.step_buffer, columns=REF_COLUMNS)
            self.episode_dfs.append(df)
            self.step_buffer = []

        return True
    
    def _on_training_end(self) -> None:
        if self.episode_dfs:
            pd.concat(self.episode_dfs, ignore_index=True).to_csv(
                self.csv_path, index=False
            )
            if self.verbose:
                print(f"📄 episode_log.csv → {self.csv_path}")

class AggregateMetricsCallback(BaseCallback):
    """
    Aggregate metrics and log them to CSV in the same format as dreamer_training_log.csv
    """
    def __init__(self, log_dir: str, step_csv_callback, log_interval: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.csv_path = os.path.join(log_dir, "ppo_training_log.csv")
        
        # Reference to the StepCSVCallback to access its data
        self.step_csv_callback = step_csv_callback
        
        # Metrics to track
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Previous log step to prevent duplicate logging
        self.last_log_step = 0
    
    def _on_step(self) -> bool:
        # Check for new episode completions
        if len(self.model.ep_info_buffer) > 0:
            # Get info from completed episodes
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info:  # This is a completed episode
                    self.episode_rewards.append(ep_info['r'])
                    self.episode_lengths.append(ep_info['l'])
        
        # Log periodically
        if self.num_timesteps >= self.last_log_step + self.log_interval:
            self.last_log_step = self.num_timesteps
            self._log_metrics()
            
        return True
    
    def _log_metrics(self):
        # Calculate averages from recent episodes
        recent_rewards = self.episode_rewards[-10:] if self.episode_rewards else [0]
        recent_lengths = self.episode_lengths[-10:] if self.episode_lengths else [0]
        
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        avg_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0
        
        # Create metrics dictionary exactly like dreamer CSV
        metrics_dict = {
            'step': self.num_timesteps,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'component_discount': 1.0,
            'component_reward': avg_reward / avg_length if avg_length > 0 else 0,
            'component_TimeLimit.truncated': 0.0
        }
        
        # Add achievement counts by accessing StepCSVCallback's data
        # This is the key fix - we get achievement data from the episode_dfs in StepCSVCallback
        for achievement in [
            "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
            "collect_sapling", "collect_stone", "collect_wood",
            "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
            "make_iron_pickaxe", "make_iron_sword",
            "make_stone_pickaxe", "make_stone_sword",
            "make_wood_pickaxe", "make_wood_sword",
            "place_furnace", "place_plant", "place_stone", "place_table",
            "wake_up"
        ]:
            # Get achievement counts from all episode data frames
            achievement_count = 0
            for df in self.step_csv_callback.episode_dfs:
                if achievement in df.columns:
                    # Sum the achievement values (1 for achieved, 0 for not)
                    achievement_count += df[achievement].sum()
            
            metrics_dict[f"achievement_{achievement}"] = achievement_count
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame([metrics_dict])
        
        # Append or create CSV
        if os.path.exists(self.csv_path):
            metrics_df.to_csv(self.csv_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(self.csv_path, index=False)
            
        if self.verbose:
            print(f"📊 Logged metrics at step {self.num_timesteps}")
            print(f"   Avg reward: {avg_reward:.2f}, Avg length: {avg_length:.1f}")
    
    def _on_training_end(self) -> None:
        # Final log at the end of training
        self._log_metrics()






class DecisionAttributionCallback(BaseCallback):
    """
    Separate CSV for sparse logging (every `log_freq` steps) if desired.
    Not strictly required now that metrics are in episode_log.csv,
    but kept for optional quick inspection.
    """

    def __init__(self, csv_path: str, log_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.log_freq = log_freq
        self.rows = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        local = self.locals
        log_prob = local["log_probs"][0].item()
        action_prob = float(np.exp(log_prob))
        entropy = -log_prob * action_prob
        value = local["values"][0].item()
        advantage = float(local.get("advantages", [0])[0])
        action = int(local["actions"][0])

        self.rows.append(
            dict(
                step_env=self.num_timesteps,
                action=action,
                action_prob=action_prob,
                entropy=entropy,
                value=value,
                advantage=advantage,
            )
        )
        return True

    def _on_training_end(self) -> None:
        if self.rows:
            pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)
            if self.verbose:
                print(f"📑 decision_attribution.csv → {self.csv_path}")


# ------------------------------------------------------------------ #
def train_ppo(total_timesteps: int = 300_000):
    log_dir = "./ppo_logs_advanced"
    ckpt_dir = "./ppo_checkpoints"
    results_dir = "./results"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    def make_env():
        env = create_environment(output_dir=results_dir)
        env = Monitor(env, os.path.join(log_dir, "ppo_monitor.csv"))
        return env
    
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)  # HWC → CHW for CnnPolicy
    
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_tb_logs")
    
    step_csv_cb = StepCSVCallback(
        csv_path=os.path.join(log_dir, "episode_log.csv")
    )
    
    ckpt_cb = CheckpointCallback(
        save_freq=10_000, save_path=ckpt_dir, name_prefix="ppo_crafter"
    )
    
    dec_attr_cb = DecisionAttributionCallback(
        csv_path=os.path.join(log_dir, "decision_attribution.csv"),
        log_freq=10_000,
    )
    
    # Pass step_csv_cb to the aggregate metrics callback
    aggregate_cb = AggregateMetricsCallback(
        log_dir=log_dir,
        step_csv_callback=step_csv_cb,  # This is the key change
        log_interval=5000, 
        verbose=1
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[step_csv_cb, ckpt_cb, dec_attr_cb, aggregate_cb],
    )
    
    model.save("ppo_crafter_model_advanced")
    print("✅ Training finished.")






if __name__ == "__main__":
    train_ppo()
