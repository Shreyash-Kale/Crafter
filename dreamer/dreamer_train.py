# dreamer_train.py - Train DreamerV2 agent and integrate with visualization
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")      # non-interactive PNG backend

from dreamer_policy import DreamerPolicy
from environment import create_environment, run_episode


def train_dreamer(
    env_name='CrafterReward-v1',
    total_steps=250000,
    log_interval=5000,
    checkpoint_dir='./dreamer_checkpoints',
    log_dir='./logs_dreamer',
    video_dir='./videos',
    load_checkpoint=False,
    num_envs=4,  # Number of parallel environments
    save_interval=10000  # Default save interval
):
    """Train a DreamerV2 agent in the Crafter environment using parallel environments."""
    
    # Create multiple environments for parallel data collection
    envs = []
    for i in range(num_envs):
        env = gym.make(env_name)
        env.seed(i)  # Set different seeds for diversity
        envs.append(env)
    
    print(f"Created {num_envs} parallel environments for data collection")
    
    # Create structured directory names based on target steps
    if load_checkpoint:
        # Extract the starting step from checkpoint_dir
        try:
            # Parse checkpoint_XXX format
            if '_' in checkpoint_dir:
                start_step = int(os.path.basename(checkpoint_dir).split('_')[-1])
            else:
                start_step = 0
        except (ValueError, IndexError):
            start_step = 0
            
        target_step = start_step + total_steps
        
        # Create new directories with step-based naming
        parent_checkpoint_dir = os.path.dirname(checkpoint_dir) if os.path.dirname(checkpoint_dir) else '.'
        parent_log_dir = os.path.dirname(log_dir) if os.path.dirname(log_dir) else '.'
        
        new_checkpoint_dir = f"{parent_checkpoint_dir}/ckpt_{target_step}"
        new_log_dir = f"{parent_log_dir}/log_{target_step}"
        
        # Create the directories
        os.makedirs(new_checkpoint_dir, exist_ok=True)
        os.makedirs(new_log_dir, exist_ok=True)
        
        print(f"Continuing training. New checkpoints will be saved to: {new_checkpoint_dir}")
        print(f"New logs will be saved to: {new_log_dir}")
        
        # Keep original dir for loading but use new dir for saving
        load_dir = checkpoint_dir
        checkpoint_dir = new_checkpoint_dir
        log_dir = new_log_dir
    else:
        # Fresh training, use step-based directories
        parent_checkpoint_dir = os.path.dirname(checkpoint_dir) if os.path.dirname(checkpoint_dir) else '.'
        parent_log_dir = os.path.dirname(log_dir) if os.path.dirname(log_dir) else '.'
        
        checkpoint_dir = f"{parent_checkpoint_dir}/ckpt_{total_steps}"
        log_dir = f"{parent_log_dir}/log_{total_steps}"
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"Starting new training. Checkpoints will be saved to: {checkpoint_dir}")
        print(f"Logs will be saved to: {log_dir}")
        
        load_dir = checkpoint_dir
    
    os.makedirs(video_dir, exist_ok=True)
    
    # Create policy with custom save interval
    policy = DreamerPolicy(
        envs[0],
        training=True,
        checkpoint_dir=checkpoint_dir,
        load_checkpoint=load_checkpoint,
        checkpoint_path=load_dir if load_checkpoint else None,
        parallel_envs=num_envs,
        training_interval=5,  # Training interval for DreamerV2
        save_interval=save_interval  # Pass save interval to policy
    )
    print("Training interval set to:", policy.training_interval)


    # Initialize step count from policy global step
    if load_checkpoint and hasattr(policy, 'global_step'):
        step_count = policy.global_step.numpy()
        print(f"Resuming training from step {step_count}")
    else:
        step_count = 0
        
    # Initialize logging
    episode_rewards = [[] for _ in range(num_envs)]
    episode_lengths = [[] for _ in range(num_envs)]
    all_env_metrics = {
        'achievements': {},
        'reward_components': {}
    }
    
    csv_path = os.path.join(log_dir, 'dreamer_training_log.csv')
    
    # Training loop
    observations = [env.reset() for env in envs]
    episode_rewards_current = [0 for _ in range(num_envs)]
    episode_lengths_current = [0 for _ in range(num_envs)]
    episode_counts = [0 for _ in range(num_envs)]
    
    print("Starting training with parallel environments...")
    
    # Create a log file
    with open(os.path.join(log_dir, 'dreamer_training.txt'), 'w') as f:
        f.write(f"Starting DreamerV2 training with {num_envs} parallel environments...\n")
        if load_checkpoint:
            f.write(f"Resuming from previous checkpoint at step {step_count}\n")
    
    
  
    
    while step_count < total_steps:
        # Get actions from policy for all environments
        actions = []
        for obs in observations:
            action = policy(obs)
            actions.append(action)


        for i, (obs, action) in enumerate(zip(observations, actions)):
            # Log decision attribution for one environment periodically
            if i == 0 and step_count % 1000 < num_envs:
                try:
                    # Ensure action is a scalar value before calling log_decision_attribution
                    action_scalar = action
                    if hasattr(action_scalar, "numpy"):  # Handle TensorFlow tensors
                        action_scalar = action_scalar.numpy()
                    if isinstance(action_scalar, np.ndarray):
                        if action_scalar.size == 1:  # Single-element array
                            action_scalar = action_scalar.item()
                        else:  # Multi-element array
                            action_scalar = int(np.argmax(action_scalar))
                            
                    # Ensure it's a plain Python int
                    action_scalar = int(action_scalar)
                    
                    # Now call with the scalar action
                    attribution = policy.log_decision_attribution(obs, action_scalar)
                    # write header once
                    attrib_path = os.path.join(log_dir, 'decision_attribution.csv')
                    write_header = not os.path.exists(attrib_path)
                    with open(attrib_path, 'a') as f:
                        if write_header:
                            f.write('step,action_taken,action_probability,world_model_score,exploration_bonus,value_estimate\n')
                        f.write(f"{step_count},"
                                f"{attribution['action_taken']},"
                                f"{attribution['action_probability']:.6f},"
                                f"{attribution['world_model_score']:.6f},"
                                f"{attribution['exploration_bonus']:.6f},"
                                f"{attribution['value_estimate']:.6f}\n")
                except Exception as e:
                    print(f"Error calculating decision attribution at step {step_count}: {e}")


        
        # Step all environments
        next_observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(envs, actions)):
            next_obs, reward, done, info = env.step(action)
            next_observations.append(next_obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            # Update episode stats
            episode_rewards_current[i] += reward
            episode_lengths_current[i] += 1
            
            # Store information about achievements and reward components
            for key, value in info.items():
                if key not in all_env_metrics['reward_components']:
                    all_env_metrics['reward_components'][key] = []
                all_env_metrics['reward_components'][key].append(value)
            
            # Handle episode termination
            if done:
                print(f"Env {i} - Episode {episode_counts[i]} - Steps: {episode_lengths_current[i]}, "
                      f"Reward: {episode_rewards_current[i]:.2f}, Total Steps: {step_count}")
                
                # Record episode statistics
                episode_rewards[i].append(episode_rewards_current[i])
                episode_lengths[i].append(episode_lengths_current[i])
                
                # Record achievements if present in info
                if 'achievements' in info:
                    for achievement, achieved in info['achievements'].items():
                        if achievement not in all_env_metrics['achievements']:
                            all_env_metrics['achievements'][achievement] = []
                        all_env_metrics['achievements'][achievement].append(achieved)
                
                # Reset for next episode
                next_observations[i] = env.reset()
                episode_rewards_current[i] = 0
                episode_lengths_current[i] = 0
                episode_counts[i] += 1
                
        # Update policy for all transitions
        for i in range(num_envs):
            policy.update(
                observations[i], 
                actions[i], 
                rewards[i], 
                dones[i], 
                next_observations[i]
            )
            
            # Reset policy state for terminated episodes
            if dones[i]:
                policy.reset()
        
        # Update current observations
        observations = next_observations
        
        # Increment step count
        step_count += num_envs
        
        # Log statistics periodically
        if step_count % log_interval < num_envs:
            # Flatten the episode rewards from all envs
            all_rewards = [reward for env_rewards in episode_rewards for reward in env_rewards[-10:]]
            all_lengths = [length for env_lengths in episode_lengths for length in env_lengths[-10:]]
            
            if all_rewards:
                avg_reward = sum(all_rewards) / len(all_rewards)
                avg_length = sum(all_lengths) / len(all_lengths)
                
                # Log to file
                with open(os.path.join(log_dir, 'dreamer_training.txt'), 'a') as f:
                    f.write(f"Step {step_count}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}\n")
                
                # Enhanced logging for reward components
                for component, values in all_env_metrics['reward_components'].items():
                    if values:
                        # Check if the values are numeric
                        numeric_values = [v for v in values[-100:] if isinstance(v, (int, float))]
                        if numeric_values:
                            avg_value = sum(numeric_values) / len(numeric_values)
                            with open(os.path.join(log_dir, 'dreamer_training.txt'), 'a') as f:
                                f.write(f"  {component}: {avg_value:.3f}\n")
                        else:
                            # For non-numeric components, just report the count
                            with open(os.path.join(log_dir, 'dreamer_training.txt'), 'a') as f:
                                f.write(f"  {component}: (non-numeric data)\n")

                
                # Save reward plot with more metrics
                plt.figure(figsize=(15, 10))
                
                # Plot episode rewards
                plt.subplot(2, 2, 1)
                for i in range(num_envs):
                    plt.plot(episode_rewards[i], alpha=0.3, label=f'Env {i}' if i == 0 else None)
                # Plot moving average
                all_rewards_flat = [r for env_r in episode_rewards for r in env_r]
                if len(all_rewards_flat) > 10:
                    window_size = min(10, len(all_rewards_flat))
                    moving_avg = [sum(all_rewards_flat[i:i+window_size])/window_size 
                                 for i in range(len(all_rewards_flat)-window_size+1)]
                    plt.plot(range(window_size-1, len(all_rewards_flat)), moving_avg, 'r-', linewidth=2, label='Moving Avg')
                plt.title('Episode Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.legend()
                
                # Plot episode lengths
                plt.subplot(2, 2, 2)
                for i in range(num_envs):
                    plt.plot(episode_lengths[i], alpha=0.3)
                plt.title('Episode Lengths')
                plt.xlabel('Episode')
                plt.ylabel('Steps')
                
                # Plot reward components if available
                if all_env_metrics['reward_components']:
                    plt.subplot(2, 2, 3)
                    for component, values in all_env_metrics['reward_components'].items():
                        if len(values) > 10:  # Only plot if we have enough data
                            # Filter for numeric values
                            numeric_values = [v for v in values if isinstance(v, (int, float))]
                            if len(numeric_values) > 10:  # Check again after filtering
                                window_size = min(100, len(numeric_values))
                                component_avg = [sum(numeric_values[i:i+window_size])/window_size 
                                                for i in range(0, len(numeric_values)-window_size+1, window_size)]
                                plt.plot(range(0, len(numeric_values)-window_size+1, window_size), 
                                        component_avg, label=component)

                    plt.title('Reward Components')
                    plt.xlabel('Steps')
                    plt.ylabel('Value')
                    plt.legend()
                
                # Plot achievements if available
                if all_env_metrics['achievements']:
                    plt.subplot(2, 2, 4)
                    achievement_counts = {}
                    for achievement, achieved in all_env_metrics['achievements'].items():
                        achievement_counts[achievement] = sum(achieved)
                    
                    if achievement_counts:
                        plt.bar(achievement_counts.keys(), achievement_counts.values())
                        plt.title('Total Achievements')
                        plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(log_dir, 'dreamer_training_metrics.png'))
                plt.close()
                
                # Save metrics to CSV
                metrics_dict = {
                    'step': step_count,
                    'avg_reward': avg_reward,
                    'avg_length': avg_length
                }
                
                # Add reward components
                for component, values in all_env_metrics['reward_components'].items():
                    if values:
                        numeric_values = [v for v in values[-100:] if isinstance(v, (int, float))]
                        if numeric_values:
                            metrics_dict[f'component_{component}'] = sum(numeric_values) / len(numeric_values)

                # Add achievement counts
                for achievement, achieved in all_env_metrics['achievements'].items():
                    if achieved:
                        metrics_dict[f'achievement_{achievement}'] = sum(achieved[-100:])
                
                # Convert to DataFrame
                metrics_df = pd.DataFrame([metrics_dict])
                
                # Append or create CSV
                if os.path.exists(csv_path):
                    metrics_df.to_csv(csv_path, mode='a', header=False, index=False)
                else:
                    metrics_df.to_csv(csv_path, index=False)
    
    print("Training complete!")
    return policy


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


def run_dreamer_episode(env_name='CrafterReward-v1', checkpoint_dir='./dreamer_checkpoints', output_dir='./logs_dreamer'):
    """Run a visualization episode with a trained DreamerV2 agent."""
    
    # Create environment
    env = create_environment(output_dir)
    
    # Create policy
    policy = DreamerPolicy(
        env,
        training=False,
        checkpoint_dir=checkpoint_dir,
        load_checkpoint=True
    )
    
    # Define policy function for run_episode
    def dreamer_policy_fn(obs):
        return policy(obs)
    
    # Run episode
    results = run_episode(env, dreamer_policy_fn)
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='DreamerV2 training for Crafter')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'visualize'],
                        help='Mode to run: train or visualize')
    parser.add_argument('--steps', type=int, default=1000000,
                        help='Total training steps')
    parser.add_argument('--checkpoint-dir', type=str, default='./dreamer_checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory for logs')
    parser.add_argument('--load-checkpoint', action='store_true',
                        help='Load existing checkpoint')
    parser.add_argument('--save-interval', type=int, default=2000,
                        help='How often to save checkpoints (steps)')
    parser.add_argument('--keep-checkpoints', type=int, default=5,
                        help='Number of recent checkpoints to keep')
    args = parser.parse_args()

    if args.mode == 'train':
        train_dreamer(
            total_steps=args.steps,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            load_checkpoint=args.load_checkpoint,
            save_interval=args.save_interval
        )
    else:
        run_dreamer_episode(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.log_dir
        )

