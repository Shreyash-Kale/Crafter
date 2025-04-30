# environment.py - Updated to use Crafter's native recorder and enhanced data logging

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import gym
import crafter
import numpy as np
import pandas as pd
import cv2
from dreamer_policy import DreamerPolicy
from pathlib import Path
import csv
import numpy as np
import imageio.v2 as iio
from pathlib import Path
import csv
import numpy as np
import imageio.v2 as iio
import os

# Configuration constants
BASE_CHECKPOINT_DIR = '/Users/sirius/Desktop/Crafter/saved_checkpoints/ckpt_'
RESULTS_DIR = './results'
DEFAULT_NUM_EPISODES = 50
DEFAULT_CHECKPOINT = 530000  # Default checkpoint number to use if not specified

import gym
import crafter

def create_environment(output_dir: str = "./results"):
    """Crafter with dense reward + recorder + safe metadata for Gym 0.25+."""
    # 1) base env with dense reward
    env = gym.make("CrafterReward-v1")

    # 2) guarantee every layer has metadata.render_modes
    def _ensure_metadata(e):
        if getattr(e, "metadata", None) is None:
            e.metadata = {}
        e.metadata.setdefault("render_modes", ["rgb_array"])
    _ensure_metadata(env)
    _ensure_metadata(env.unwrapped)          # reach the real core env as well

    # 3) wrap with Crafter’s recorder
    env = crafter.Recorder(
        env,
        output_dir,
        save_stats=True,
        save_episode=True,
        save_video=True,
    )

    # 4) recorder itself needs the key too
    _ensure_metadata(env)

    return env


def run_episode(
    env,
    policy_fn,
    max_frames: int = 1_000,
    output_dir: str = "./results",
    episode_id: int = 0,
    record_video: bool = True,
):
    """
    Execute ONE episode, saving
    • episode_.csv – per-timestep log (always)
    • episode_.mp4 – RGB video (optional)
    
    The CSV contains (i) basic transition data, (ii) flattened achievements,
    and (iii) the **decision–attribution** fields returned by the policy.
    
    If an attribution key is missing at a timestep it is filled with `None`
    so the header is *always* present and the column lengths stay equal.
    
    Parameters
    ----------
    env : gym.Env
    policy_fn : callable
        Signature: action = policy_fn(obs)
        If the *object* behind `policy_fn` implements
        `.decision_attribution(obs) -> dict`
        those values are stored as explainability signals.
    max_frames : int
    output_dir : str
    episode_id : int
    record_video : bool
    """

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract checkpoint number and episode number from directory structure
    out_dir_str = str(out_dir)
    parts = out_dir_str.split(os.sep)
    
    # Look for checkpoint and episode in the path
    checkpoint_num = None
    for part in parts:
        if part.startswith('checkpoint_'):
            checkpoint_num = part.split('_')[1]
    
    # Get episode number from the episode_id or directory name
    if 'episode_' in out_dir_str:
        for part in parts:
            if part.startswith('episode_'):
                episode_num = part.split('_')[1]
    else:
        episode_num = f"{episode_id:03d}"
    
    # Create standardized base filename
    if checkpoint_num:
        base_filename = f"ckpt{checkpoint_num}_episode{episode_num}"
    else:
        # Fallback to old naming if we can't extract the checkpoint number
        base_filename = f"episode_{episode_num}"
    
    # Use this base filename for all output files
    csv_path = out_dir / f"{base_filename}.csv"
    video_out = out_dir / f"{base_filename}.mp4"
    
    # Resolve the policy *object* so we can look for .decision_attribution
    policy_obj = getattr(policy_fn, "__self__", policy_fn)
    
    # Attribution keys we want in every row
    ATTR_KEYS = [
        "logit",
        "action_probability",
        "value_estimate",
        "exploration_bonus",
        "world_model_score",
    ]
    

    rows, video_frames = [], []
    reset_result = env.reset()  # Update this line
    if isinstance(reset_result, tuple):  # Handle gym versions returning (obs, info)
        obs, info = reset_result
    else:  # Handle gym versions returning only obs
        obs, info = reset_result, {}
    
    cumulative_r = 0.0
    for t in range(max_frames):
        # 1) Action and (optional) attribution
        action = policy_fn(obs)  # --- a_t
        attrib = {k: None for k in ATTR_KEYS}  # default Nones
        if hasattr(policy_obj, "decision_attribution"):
            try:
                raw_attr = policy_obj.decision_attribution(obs)
                if raw_attr:  # may be empty dict
                    for k in ATTR_KEYS:
                        if k in raw_attr:
                            attrib[k] = raw_attr[k]
            except Exception as e:  # robust to any error
                print(f"⚠️ attribution error @t={t}: {e}")
        else:
            if t == 0:  # warn only once
                print("⚠️ Policy has no decision-attribution method")
        
        # 2) Environment step
        step_result = env.step(action)
        
        if len(step_result) == 5:  # Newer gym versions
            next_obs, r, terminated, truncated, step_info = step_result
            done = terminated or truncated
        else:  # Older gym versions
            next_obs, r, done, step_info = step_result
            

        cumulative_r += r
        
        # 3) Optional video frame
        if record_video and hasattr(env, "render"):
            try:
                frame = env.render(mode="rgb_array")
                video_frames.append(frame)
            except Exception:
                pass  # rendering can fail on headless servers
        
        # 4) Collect row 
        row = dict(
            time_step=t,
            action=int(action) if np.isscalar(action) else action,
            reward=float(r),
            cumulative_reward=float(cumulative_r),
            done=bool(done),
        )
        
        # Flatten achievements (if present)
        if "achievements" in step_info:
            for ach, flag in step_info["achievements"].items():
                row[f"{ach}"] = int(flag)
        
        # Add inventory and stats if available
        # Define all inventory items we want to track
        inventory_items = [
            "sapling", "wood", "stone", "coal", "iron", "diamond",
            "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
            "wood_sword", "stone_sword", "iron_sword"
        ]
        
        # Add inventory items to the row (initialize to 0 if not present)
        if "inventory" in step_info:
            for item, amount in step_info["inventory"].items():
                row[item] = int(amount)
            
            # Ensure all inventory items are present (even if 0)
            for item in inventory_items:
                if item not in row:
                    row[item] = 0
        elif hasattr(env, "state"):
            # Try to access inventory from env.state (used in some versions of Crafter)
            for item in inventory_items:
                if hasattr(env.state, item):
                    row[item] = getattr(env.state, item)
                else:
                    row[item] = 0
        else:
            # Initialize all inventory items to 0 if not found
            for item in inventory_items:
                row[item] = 0
        
        # Add player stats if available
        stats = ["health", "food", "drink", "energy"]
        for stat in stats:
            if stat in step_info:
                row[stat] = step_info[stat]
            elif hasattr(env, "state") and hasattr(env.state, stat):
                row[stat] = getattr(env.state, stat)
            elif stat not in row:  # Ensure the stat is included even if not found
                row[stat] = 9  # Default starting value in Crafter
        
        # Merge attribution
        row.update(attrib)
        rows.append(row)
        
        # 5) Advance
        obs = next_obs
        if done:
            break
    

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    

    if record_video and video_frames:
        iio.mimsave(video_out, video_frames, fps=15)
    
    print(f"Episode completed with total reward: {cumulative_r:.2f}")
    print(f"Results saved in {csv_path.parent}")
    
    return cumulative_r, csv_path



def find_latest_checkpoint(checkpoint_dir):
    """Helper function to find the latest checkpoint number."""
    import os
    import re
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('ckpt-') and f.endswith('.index')]
    if not checkpoint_files:
        return None
    
    checkpoint_numbers = [int(re.search(r'ckpt-(\d+)', f).group(1)) for f in checkpoint_files]
    return max(checkpoint_numbers)



def run_with_dreamer(checkpoint_number=90000, checkpoint=None, num_episodes=DEFAULT_NUM_EPISODES, results_dir=RESULTS_DIR):
    """Run multiple episodes with DreamerV2 agent from a specific checkpoint"""
    # Construct the full checkpoint directory path
    checkpoint_dir = f"{BASE_CHECKPOINT_DIR}{checkpoint_number}"
    
    # Find the latest checkpoint if not specified
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(checkpoint_dir)
        
    if checkpoint is None:
        print(f"No checkpoints found in {checkpoint_dir}. Cannot run evaluation.")
        return None
        
    print(f"Using checkpoint: {checkpoint} from {checkpoint_dir}")
    
    # Create algorithm directory structure
    algo_dir = os.path.join(results_dir, 'dreamer_v2')
    checkpoint_dir_output = os.path.join(algo_dir, f'checkpoint_{checkpoint}')
    os.makedirs(checkpoint_dir_output, exist_ok=True)
    
    # Create environment with Crafter's recorder for each episode
    results = []
    for episode in range(1, num_episodes + 1):
        # Create episode-specific directory
        episode_dir = os.path.join(checkpoint_dir_output, f'episode_{episode:03d}')
        os.makedirs(episode_dir, exist_ok=True)
        
        # Create environment with recorder pointing to this episode's directory
        env = create_environment(episode_dir)
        
        # Create agent with checkpoint
        agent = DreamerPolicy(
            env,
            training=False,
            checkpoint_dir=checkpoint_dir,
            load_checkpoint=True,
            checkpoint_number=checkpoint
        )
        

        def policy_func(obs):
            return agent(obs)
        policy_func.__self__ = agent  # Attach the agent as the __self__ attribute

        # Run episode
        total_reward, csv_path = run_episode(env, policy_func, output_dir=episode_dir)  # Unpack the tuple
        results.append((total_reward, csv_path))  # Append as a tuple
        
        # Print the total reward
        print(f"Completed episode {episode}/{num_episodes} with reward: {total_reward:.2f}")

    
    return results





if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run evaluation episodes with DreamerV2")
    parser.add_argument("--checkpoint-number", type=int, default=DEFAULT_CHECKPOINT,
                        help="Checkpoint folder number (e.g., 450000 for ckpt_450000)")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES,
                        help="Number of episodes to run")
    args = parser.parse_args()
    
    # Run evaluation
    checkpoint_number = args.checkpoint_number
    latest_checkpoint = find_latest_checkpoint(f"{BASE_CHECKPOINT_DIR}{checkpoint_number}")
    
    if latest_checkpoint:
        print(f"Running evaluation with checkpoint {latest_checkpoint} from folder ckpt_{checkpoint_number}")
        results = run_with_dreamer(
            checkpoint_number=checkpoint_number,
            checkpoint=latest_checkpoint,
            num_episodes=args.num_episodes
        )
        
        print(f"Evaluation complete! Results saved in {RESULTS_DIR}/dreamer_v2/checkpoint_{latest_checkpoint}/")
    else:
        print(f"No checkpoints found in {BASE_CHECKPOINT_DIR}{checkpoint_number}. Train a model first.")

