import os, json, csv, numpy as np
from tqdm import trange
import torch
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from ppo_environment import create_environment # your wrapper


SAVED_CHECKPOINT = 270000
MODEL_PATH   = f"/Users/sirius/Desktop/Workspace/Og_Crafter/PPO_Code/ppo_checkpoints/ppo_crafter_{SAVED_CHECKPOINT}_steps"
N_EPISODES   = 300                       # how many episodes to run
RESULTS_DIR  = f"./ppo/ppo_eval_results{SAVED_CHECKPOINT}"   # root folder
SAVE_VIDEO   = True                      # False → faster, no mp4
# -------------------------------------------------------------------------

# Create both RESULTS_DIR **and** its eval_run sub-folder up-front
EVAL_DIR = os.path.join(RESULTS_DIR, f"eval_run_{N_EPISODES}")
os.makedirs(EVAL_DIR, exist_ok=True)

# 1. Build evaluation environment --------------------------------------
def make_env():
    # video+stats saved inside RESULTS_DIR/eval_run/
    return create_environment(output_dir=EVAL_DIR,
                             save_video=SAVE_VIDEO)

eval_env = VecTransposeImage(DummyVecEnv([make_env])) # HWC → CHW

# 2. Load the trained PPO policy ---------------------------------------
model = PPO.load(MODEL_PATH, env=eval_env)
print(f"Loaded model from {MODEL_PATH}")

# 3. Detailed per-episode logging --------------------------------------
ep_rewards, ep_lengths, ep_achievements, ep_health = [], [], [], []

for ep in trange(N_EPISODES, desc="Evaluating"):
    obs = eval_env.reset()
    done = False
    total_r, steps, ach = 0.0, 0, {}
    
    # Create a list to store all step data for this episode
    episode_data = []
    
    while not done:
        # Get action
        action, _ = model.predict(obs, deterministic=True)
        
        # Extract decision attributes from the policy
        with torch.no_grad():
            # Convert observation to tensor and ensure it's float32
            obs_tensor = torch.as_tensor(obs).to(model.policy.device).float()
            
            # Get distribution and value using public API methods
            distribution = model.policy.get_distribution(obs_tensor)
            values = model.policy.predict_values(obs_tensor)
            
            # Calculate decision attributes
            action_tensor = torch.as_tensor([action[0]]).long().to(model.policy.device)
            log_prob = distribution.log_prob(action_tensor)
            action_prob = float(torch.exp(log_prob).cpu().numpy())
            entropy = float(distribution.entropy().cpu().numpy())
            value = float(values.cpu().numpy()[0])
        
        # Execute action
        obs, reward, done, info = eval_env.step(action)
        total_r += reward[0]
        steps += 1
        
        # Track achievements
        for k, v in info[0].get("achievements", {}).items():
            ach[k] = ach.get(k, False) or v
        
        # Store step data including decision attributes
        step_data = {
            "time_step": steps - 1,
            "action": int(action[0]),
            "reward": float(reward[0]),
            "cumulative_reward": total_r,
            "action_probability": action_prob,
            "entropy": entropy,
            "value": value,
            "advantage": 0.0, # Cannot calculate during evaluation
            'health': info[0].get('health', -1)
        }
        
        # Add stats and inventory
        for key in ["health", "food", "drink", "energy"]:
            if key in info[0]:
                step_data[key] = info[0][key]
        
        inv = info[0].get("inventory", {})
        for res in ["sapling", "wood", "stone", "coal", "iron", "diamond",
                   "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
                   "wood_sword", "stone_sword", "iron_sword"]:
            step_data[res] = inv.get(res, 0)
        
        # Add achievements
        for a in ach.keys():
            step_data[a] = int(ach.get(a, False))
        
        episode_data.append(step_data)
    
    ep_rewards.append(total_r)
    ep_lengths.append(steps)
    ep_achievements.append(ach)
    ep_health.append(int(info[0].get("health", 0)))
    
    # Write individual CSV for this episode with decision attributes
    episode_csv_path = os.path.join(RESULTS_DIR, "eval_run", f"episode_{ep}.csv")
    pd.DataFrame(episode_data).to_csv(episode_csv_path, index=False)
    print(f"Episode {ep} data saved to {episode_csv_path}")

# Calculate and print summary statistics
mean_reward = np.mean(ep_rewards)
std_reward = np.std(ep_rewards)
print(f"Mean reward over {N_EPISODES} episodes: {mean_reward:.2f} ± {std_reward:.2f}")
