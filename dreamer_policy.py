# dreamer_policy.py - Integration of DreamerV2 with the Crafter environment

import os
import numpy as np
from dreamerv2 import DreamerV2
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class EnhancedReplayBuffer:
    """Enhanced replay buffer with episode tracking and efficient sampling."""
    
    def __init__(self, capacity, observation_shape, action_dim, discrete=True):
        self.capacity = capacity
        self.discrete = discrete
        
        # Initialize buffers
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        if discrete:
            self.actions = np.zeros((capacity,), dtype=np.int32)
        else:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=bool)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        
        # Add episode tracking for more efficient sampling
        self.episode_ends = []  # List of episode end indices
        self.current_episode_start = 0
        
        self.idx = 0
        self.full = False
        
        # Priority sampling
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_increment = 0.001  # Beta increment per sampling
        self.epsilon = 1e-5  # Small constant to avoid zero priority
        
    def add(self, obs, action, reward, done, next_obs):
        """Add a transition to the buffer."""
        # Store transition
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.next_observations[self.idx] = next_obs
        
        # Set priority to maximum for new transitions
        max_priority = np.max(self.priorities) if self.idx > 0 else 1.0
        self.priorities[self.idx] = max_priority
        
        # Track episode boundaries
        if done:
            self.episode_ends.append(self.idx)
            self.current_episode_start = (self.idx + 1) % self.capacity
        
        # Update index
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True
    
    def _get_valid_sequence_indices(self, sequence_length):
        """Get valid starting indices for sequences of the given length."""
        valid_indices = []
        if self.full:
            valid_size = self.capacity
        else:
            valid_size = self.idx
        
        # Find episode boundaries to avoid crossing them
        episode_boundaries = set()
        for end_idx in self.episode_ends:
            if end_idx < valid_size - sequence_length:
                for i in range(sequence_length):
                    episode_boundaries.add((end_idx + i) % self.capacity)
        
        # Collect valid starting points
        for i in range(valid_size - sequence_length):
            # Check if any position in the sequence is an episode boundary
            is_valid = True
            for j in range(sequence_length):
                if (i + j) % self.capacity in episode_boundaries:
                    is_valid = False
                    break
            
            if is_valid:
                valid_indices.append(i)
                
        return valid_indices
    
    def sample(self, batch_size, sequence_length):
        """Sample a batch of sequences with prioritized sampling."""
        # Get valid starting indices
        valid_indices = self._get_valid_sequence_indices(sequence_length)
        if not valid_indices:
            raise ValueError("Not enough valid sequences in buffer for sampling")
        
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        probs = self.priorities[valid_indices] ** self.alpha
        probs /= np.sum(probs)
        
        # Sample indices from valid_indices, not from the entire buffer range
        # THIS IS THE FIX - sample from the valid_indices array itself
        idx_of_indices = np.random.choice(len(valid_indices), size=batch_size, p=probs)
        indices = [valid_indices[i] for i in idx_of_indices]
        
        # Calculate importance sampling weights
        weights = (1.0 / (len(valid_indices) * probs[idx_of_indices])) ** self.beta
        weights /= np.max(weights)  # Normalize weights
        
        # Gather sequences
        obs_seq = np.array([self.observations[i:i+sequence_length] for i in indices])
        action_seq = np.array([self.actions[i:i+sequence_length] for i in indices])
        reward_seq = np.array([self.rewards[i:i+sequence_length] for i in indices])
        done_seq = np.array([self.dones[i:i+sequence_length] for i in indices])
        next_obs_seq = np.array([self.next_observations[i:i+sequence_length] for i in indices])
        
        return obs_seq, action_seq, reward_seq, done_seq, next_obs_seq, weights, indices

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self):
        """Return the current size of the buffer."""
        if self.full:
            return self.capacity
        else:
            return self.idx



class DreamerPolicy:
    """Policy implementation using DreamerV2 for the Crafter environment."""
    
    def __init__(
        self,
        env,
        training=True,
        replay_capacity=100000,
        batch_size=50,
        sequence_length=50,
        training_interval=10,
        save_interval=1000,
        checkpoint_dir='./dreamer_checkpoints',
        load_checkpoint=False,
        parallel_envs=1,
        checkpoint_number=None,
        checkpoint_path=None
    ):
        self.env = env
        self.training = training
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.training_interval = training_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = checkpoint_path or checkpoint_dir  # Path to load from
    
        # Create replay buffer
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
        self.replay_buffer = EnhancedReplayBuffer(
            replay_capacity,
            env.observation_space.shape,
            action_dim,
            discrete=hasattr(env.action_space, 'n')
    )
        
        self.parallel_envs = parallel_envs
        
        # Create DreamerV2 agent
        self.agent = DreamerV2(
            observation_space=env.observation_space,
            action_space=env.action_space,
            checkpoint_dir=checkpoint_dir,
            load_checkpoint=load_checkpoint,
            actor_entropy=1e-3,          # ← more exploration
            imagination_horizon=25       # ← longer rollouts
        )
        
        # Add global step tracking
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
        
        # When loading a checkpoint
        if load_checkpoint:
            if checkpoint_number is not None:
                checkpoint_path = os.path.join(self.checkpoint_path, f'ckpt-{checkpoint_number}')
                status = self.agent.checkpoint.restore(checkpoint_path)
                status.expect_partial()
                # Set global_step to the loaded checkpoint number
                self.global_step.assign(checkpoint_number)
                print(f"Loaded checkpoint from {checkpoint_path}, resuming from step {checkpoint_number}")
            else:
                # Try to load the latest checkpoint
                latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                if latest_checkpoint:
                    status = self.agent.checkpoint.restore(latest_checkpoint)
                    status.expect_partial()
                    # Extract step number from checkpoint name
                    step = int(latest_checkpoint.split('-')[-1])
                    self.global_step.assign(step)
                    print(f"Loaded latest checkpoint: {latest_checkpoint}, resuming from step {step}")
                else:
                    print(f"No checkpoint found at {self.checkpoint_path}")
        
        # Initialize state
        self.current_obs = None
        self.step_count = self.global_step.numpy()  # Initialize step count from global step



    @property
    def recurrent_state(self):
        return self.agent.recurrent_state

    @property
    def encoder(self):
        return self.agent.encoder

    @property
    def discrete_state(self):
        return self.agent.discrete_state

    def _forward(self, obs):
        """
        Build the SAME 2048-D feature vector used during training:
            1024 (recurrent)  +  1024 (flattened discrete)
        so the critic is *always* created with the right input size.
        """
        if self.agent.recurrent_state is None or self.agent.discrete_state is None:
            self.agent.init_state()

        rec  = self.agent.recurrent_state                            # (1,1024)
        disc = tf.cast(tf.reshape(self.agent.discrete_state, [1, -1]),
                       tf.float32)                                    # (1,1024)
        model_feat = tf.concat([rec, disc], axis=-1)                  # (1,2048)

        logits = self.agent.actor(model_feat)                         # (1,A)
        value  = self.agent.critic(model_feat)                        # (1,1)
        return model_feat[0], logits[0], value[0, 0]

   
   
    # Simple  wrappers that just forward the call to DreamerV2

    def log_decision_attribution(self, obs, action):
        """
        Delegate to the agent's implementation of this method
        """
        # Route to the agent implementation instead of trying to implement it here
        return self.agent.log_decision_attribution(obs, action)



    def decision_attribution(self, obs):
        """
        1. Ask the policy for the action it would take on `obs`
        2. Convert that result to a *scalar int* (handles Tensor / ndarray / int)
        3. Hand both obs and the clean int to DreamerV2 for the real attribution
        """
        # print(f"DEBUG: In decision_attribution - observation shape: {obs.shape}")
        
        try:
            raw = self(obs)  # may be Tensor, ndarray, or int
            # print(f"DEBUG: Raw action from policy: type={type(raw)}, value={raw}")
            
            if hasattr(raw, "numpy"):  # tf.Tensor ➜ ndarray
                raw = raw.numpy()
                # print(f"DEBUG: After numpy conversion: type={type(raw)}, value={raw}")
                
            if isinstance(raw, np.ndarray):
                if raw.size == 1:  # e.g. array([3])
                    raw = raw.item()
                else:  # e.g. one-hot or logits → pick arg-max
                    raw = int(np.argmax(raw))
                # print(f"DEBUG: After array handling: type={type(raw)}, value={raw}")
                
            action_int = int(raw)  # final guarantee
            # print(f"DEBUG: Final action_int: {action_int}")
            
            return self.agent.log_decision_attribution(obs, action_int)
        except Exception as e:
            import traceback
            print(f"ERROR in decision_attribution: {e}")
            print(traceback.format_exc())
            return None  # Return None instead of raising an exception




    def load(self, checkpoint_path):
        """Load weights from a checkpoint."""
        # The agent's checkpoint is managed by the agent.ckpt object
        status = self.agent.ckpt.restore(checkpoint_path)
        # Optional: Use expect_partial() to silence warnings about optimizer variables
        status.expect_partial()
        print(f"Loaded checkpoint: {checkpoint_path}")


    def reset(self):
        """Reset the policy state."""
        self.agent.init_state()
    
    
    def __call__(self, obs):
        """Generate an action based on observation."""
        # Store current observation
        self.current_obs = obs
        
        # Get action and updated states from DreamerV2 agent
        action_tensor, recurrent_state, discrete_state = self.agent.policy(
            obs, 
            recurrent_state=self.agent.recurrent_state,
            discrete_state=self.agent.discrete_state,
            training=self.training
        )
        
        # Update the agent's state for next time
        self.agent.recurrent_state = recurrent_state
        self.agent.discrete_state = discrete_state
        
        # Convert to numpy for environment interaction
        action_np = action_tensor.numpy()
        
        # For Crafter environment, convert to a simple integer
        if self.agent.discrete_actions:
            # Handle different possible shapes
            if isinstance(action_np, np.ndarray):
                if action_np.size == 1:
                    action_np = action_np.item()  # Convert single-element array to scalar
                else:
                    action_np = np.argmax(action_np)  # Get the index of max value
        
        return action_np

    
    def update(self, obs, action, reward, done, next_obs):
        """Update policy with new transition."""
        # Add to replay buffer
        self.replay_buffer.add(obs, action, reward, done, next_obs)
        
        # Increment step count
        self.step_count += 1
        
        # Train on batches at regular intervals
        if self.training and len(self.replay_buffer) > self.batch_size * self.sequence_length and self.step_count % self.training_interval == 0:
            self.train_batch()
        
        # Save at regular intervals
        if self.training and self.step_count % self.save_interval == 0:
            self.agent.save(self.step_count)
    
    def train_batch(self):
        """Train on a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough data yet
        
        # Sample from replay buffer, now with prioritized replay
        obs, actions, rewards, dones, next_obs, weights, indices = self.replay_buffer.sample(
            self.batch_size, self.sequence_length
        )
        
        # Train the agent
        metrics = self.agent.train_batch(obs, actions, rewards, dones, next_obs)
        
        # Update priorities in the replay buffer based on training loss
        if hasattr(self.replay_buffer, 'update_priorities'):
            # Use model loss as a simple priority metric
            priorities = np.ones_like(indices) * max(0.1, metrics.get('model/loss', 1.0))
            self.replay_buffer.update_priorities(indices, priorities)

        # Log metrics to TensorBoard
        self.agent.log_metrics(metrics)
        
        return metrics

    def log_latent_state(self):
        """Log the current latent state for explainability"""
        if not hasattr(self.agent, 'recurrent_state') or not hasattr(self.agent, 'discrete_state'):
            return None
            
        if self.agent.recurrent_state is None or self.agent.discrete_state is None:
            return None
        
        # Convert tensors to numpy arrays
        recurrent_state = self.agent.recurrent_state.numpy()
        discrete_state = self.agent.discrete_state.numpy()
        
        # Reshape discrete state for easier analysis
        discrete_reshaped = discrete_state.reshape(-1)
        
        # Create features by combining recurrent and discrete states
        combined_features = np.concatenate([recurrent_state[0], discrete_reshaped])
        
        # Extract information about discrete state
        discrete_argmax = np.argmax(discrete_state.reshape(
            self.agent.discrete_size, self.agent.discrete_classes), axis=1)
        
        # Return state representation
        return {
            'recurrent_avg': float(np.mean(recurrent_state)),
            'recurrent_std': float(np.std(recurrent_state)),
            'feature_range': float(np.max(combined_features) - np.min(combined_features)),
            'discrete_categories': discrete_argmax.tolist(),
            'recurrent_state': recurrent_state.tolist()[0][:10],
            'discrete_state_flat': discrete_reshaped.tolist()[:10]
        }
