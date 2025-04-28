# dreamerv2.py - Implementation of DreamerV2 algorithm for Crafter

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import datetime
from collections import deque

tfd = tfp.distributions

class RSSM(tf.keras.Model):
    """Recurrent State-Space Model with discrete latent variables"""
    
    def __init__(
        self, 
        embedding_size=1024, 
        recurrent_state_size=1024,
        discrete_size=32,  # Number of categorical variables
        discrete_classes=32,  # Number of classes per categorical variable
        hidden_size=1024,
        activation=tf.nn.elu
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.recurrent_state_size = recurrent_state_size
        self.discrete_size = discrete_size
        self.discrete_classes = discrete_classes
        self.hidden_size = hidden_size
        self.activation = activation
        
        # Define model components
        self.gru_cell = tf.keras.layers.GRUCell(self.recurrent_state_size)
        
        # Prior networks
        self.prior_dense = tf.keras.layers.Dense(self.hidden_size, activation=self.activation)
        self.prior_discrete_logits = tf.keras.layers.Dense(self.discrete_size * self.discrete_classes)
        
        # Posterior networks
        self.posterior_dense1 = tf.keras.layers.Dense(self.hidden_size, activation=self.activation)
        self.posterior_dense2 = tf.keras.layers.Dense(self.hidden_size, activation=self.activation)
        self.posterior_discrete_logits = tf.keras.layers.Dense(self.discrete_size * self.discrete_classes)
        
    def initial_state(self, batch_size):
        """Initialize the recurrent and discrete states."""
        return (
            tf.zeros([batch_size, self.recurrent_state_size], dtype=tf.float32),
            tf.zeros([batch_size, self.discrete_size, self.discrete_classes], dtype=tf.float32)
        )
        
    def observe(self, embed, action, prev_recurrent_state, prev_discrete_state):
        """Update the model state based on an observation and action."""
        # Ensure consistent types
        embed = tf.cast(embed, tf.float32)
        action = tf.cast(action, tf.float32)
        prev_recurrent_state = tf.cast(prev_recurrent_state, tf.float32)
        prev_discrete_state = tf.cast(prev_discrete_state, tf.float32)
        
        # Concatenate previous discrete state and action
        discrete_flat = tf.reshape(prev_discrete_state,
                                [-1, self.discrete_size * self.discrete_classes])
        gru_input = tf.concat([
            discrete_flat,
            action
        ], axis=-1)
        
        # Update recurrent state
        recurrent_state, _ = self.gru_cell(gru_input, [prev_recurrent_state])
        
        # Compute prior
        prior_hidden = self.prior_dense(recurrent_state)
        prior_discrete_logits = self.prior_discrete_logits(prior_hidden)
        prior_discrete_logits = tf.reshape(prior_discrete_logits,
                                        [-1, self.discrete_size, self.discrete_classes])
        prior_discrete_dist = tfd.Independent(
            tfd.OneHotCategorical(logits=prior_discrete_logits), 1)
        
        # Compute posterior with observation
        posterior_input = tf.concat([recurrent_state, embed], axis=-1)
        posterior_hidden1 = self.posterior_dense1(posterior_input)
        posterior_hidden2 = self.posterior_dense2(posterior_hidden1)
        posterior_discrete_logits = self.posterior_discrete_logits(posterior_hidden2)
        posterior_discrete_logits = tf.reshape(posterior_discrete_logits,
                                            [-1, self.discrete_size, self.discrete_classes])
        posterior_discrete_dist = tfd.Independent(
            tfd.OneHotCategorical(logits=posterior_discrete_logits), 1)
        
        # Sample from posterior
        discrete_state = posterior_discrete_dist.sample()
        
        return (
            recurrent_state,
            discrete_state,
            prior_discrete_dist,
            posterior_discrete_dist
        )

    def imagine(self, action, recurrent_state, discrete_state):
        """Imagine the next state given the current state and action."""
        # Prepare action as input
        action = tf.cast(action, tf.float32)
        
        # Concatenate previous discrete state and action
        discrete_flat = tf.reshape(discrete_state, 
                                   [-1, self.discrete_size * self.discrete_classes])
        gru_input = tf.concat([
                    tf.cast(discrete_flat, tf.float32), 
                    tf.cast(action, tf.float32)
                ], axis=-1)
        
        # Update recurrent state
        recurrent_state, _ = self.gru_cell(gru_input, [recurrent_state])
        
        # Compute prior (which becomes the next state in imagination)
        prior_hidden = self.prior_dense(recurrent_state)
        prior_discrete_logits = self.prior_discrete_logits(prior_hidden)
        prior_discrete_logits = tf.reshape(prior_discrete_logits, 
                                          [-1, self.discrete_size, self.discrete_classes])
        prior_discrete_dist = tfd.Independent(
            tfd.OneHotCategorical(logits=prior_discrete_logits), 1)
        
        # Sample from prior
        discrete_state = prior_discrete_dist.sample()
        
        return recurrent_state, discrete_state, prior_discrete_dist


class Encoder(tf.keras.Model):
    """Encoder for DreamerV2."""
    
    def __init__(
        self, 
        embedding_size=1024,
        activation=tf.nn.elu
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.activation = activation
        
        # Define CNN layers
        self.conv1 = tf.keras.layers.Conv2D(32, 4, strides=2, padding='VALID', activation=self.activation)
        self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=2, padding='VALID', activation=self.activation)
        self.conv3 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='VALID', activation=self.activation)
        self.conv4 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='VALID', activation=self.activation)
        self.dense = tf.keras.layers.Dense(self.embedding_size, activation=self.activation)

    def call(self, obs):
        """Encode the observation into a latent embedding."""
        # Normalize pixel values
        obs = tf.cast(obs, tf.float32) / 255.0
        
        # Apply CNN layers
        x = self.conv1(obs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten and apply dense layer
        x = tf.reshape(x, [obs.shape[0], -1])
        embedding = self.dense(x)
        
        return embedding


class Decoder(tf.keras.Model):
    """Decoder for DreamerV2."""
    
    def __init__(
        self,
        output_shape,
        activation=tf.nn.elu
    ):
        super().__init__()
        self.target_shape = output_shape
        self.activation = activation
        
        # Calculate appropriate size for upsampling
        # For 64x64 inputs, we need to start with 4x4 feature maps
        self.dense1 = tf.keras.layers.Dense(1024, activation=self.activation)
        self.dense2 = tf.keras.layers.Dense(4 * 4 * 128, activation=self.activation)
        
        # Adjust deconvolution to match 64x64 output size (4->8->16->32->64)
        self.deconv1 = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='SAME', activation=self.activation)
        self.deconv2 = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='SAME', activation=self.activation)
        self.deconv3 = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='SAME', activation=self.activation)
        self.deconv4 = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='SAME', activation=None)
        
    def call(self, features):
        """Decode a latent state to an observation."""
        # Apply dense layers
        x = self.dense1(features)
        x = self.dense2(x)
        
        # Reshape for deconvolution - using 4x4 as base size for 64x64 output
        x = tf.reshape(x, [-1, 4, 4, 128])
        
        # Apply deconvolution layers (4->8->16->32->64)
        x = self.deconv1(x)  # 8x8
        x = self.deconv2(x)  # 16x16
        x = self.deconv3(x)  # 32x32
        x = self.deconv4(x)  # 64x64
        
        # Scale output to [0, 255]
        mean = (x + 0.5) * 255.0
        
        # Clip to valid range
        mean = tf.clip_by_value(mean, 0, 255)
        
        # Create distribution
        return tfd.Independent(tfd.Normal(mean, 1.0), 3)


class DenseDecoder(tf.keras.Model):
    """Dense decoder for reward and value prediction."""
    
    def __init__(
        self,
        output_shape,
        hidden_size=400,
        num_layers=2,
        activation=tf.nn.elu,
        dist='normal'
    ):
        super().__init__()
        self.target_shape = output_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.dist = dist
        
        # Define dense layers
        self.hidden_layers = []
        for _ in range(self.num_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(self.hidden_size, activation=self.activation))
        
        # Output layers
        if self.dist == 'normal':
            self.mean_layer = tf.keras.layers.Dense(np.prod(self.target_shape))
            self.std_layer = tf.keras.layers.Dense(np.prod(self.target_shape))
        elif self.dist == 'binary':
            self.logit_layer = tf.keras.layers.Dense(np.prod(self.target_shape))
        elif self.dist == 'categorical':
            self.logit_layer = tf.keras.layers.Dense(np.prod(self.target_shape))
        
    def call(self, features):
        """Predict from a latent state."""
        # Apply hidden layers
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Apply output layers
        if self.dist == 'normal':
            mean = self.mean_layer(x)
            std = self.std_layer(x)
            std = tf.nn.softplus(std) + 0.1
            
            # Reshape if needed
            if len(self.target_shape) > 1:
                mean = tf.reshape(mean, [-1] + self.target_shape)
                std = tf.reshape(std, [-1] + self.target_shape)
                
            return tfd.Independent(tfd.Normal(mean, std), len(self.target_shape))
        
        elif self.dist == 'binary':
            logits = self.logit_layer(x)
            
            # Reshape if needed
            if len(self.target_shape) > 1:
                logits = tf.reshape(logits, [-1] + self.target_shape)
                
            return tfd.Independent(tfd.Bernoulli(logits=logits), len(self.target_shape))
        
        elif self.dist == 'categorical':
            logits = self.logit_layer(x)
            
            # Reshape if needed
            if len(self.target_shape) > 1:
                logits = tf.reshape(logits, [-1] + self.target_shape[:-1] + [1])
                
            return tfd.Independent(tfd.OneHotCategorical(logits=logits), len(self.target_shape) - 1)


class Actor(tf.keras.Model):
    """Actor network for DreamerV2."""
    
    def __init__(
        self,
        action_size,
        hidden_size=400,
        num_layers=4,
        activation=tf.nn.elu,
        discrete=True
    ):
        super().__init__()
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.discrete = discrete
        
        # Define hidden layers
        self.hidden_layers = []
        for _ in range(self.num_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(self.hidden_size, activation=self.activation))
        
        # Output layer
        if self.discrete:
            self.logit_layer = tf.keras.layers.Dense(self.action_size)
        else:
            self.mean_layer = tf.keras.layers.Dense(self.action_size)
            self.std_layer = tf.keras.layers.Dense(self.action_size)
        
    def call(self, features):
        """Predict action distribution from a latent state."""
        # Apply hidden layers
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Apply output layers
        if self.discrete:
            logits = self.logit_layer(x)
            return tfd.OneHotCategorical(logits=logits)
        else:
            mean = self.mean_layer(x)
            std = self.std_layer(x)
            std = tf.nn.softplus(std) + 0.1
            return tfd.Independent(tfd.Normal(mean, std), 1)


class Critic(tf.keras.Model):
    """Critic network for DreamerV2."""
    
    def __init__(
        self,
        hidden_size=400,
        num_layers=4,
        activation=tf.nn.elu
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        
        # Define hidden layers
        self.hidden_layers = []
        for _ in range(self.num_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(self.hidden_size, activation=self.activation))
        
        # Output layer
        self.value_layer = tf.keras.layers.Dense(1)
        
    def call(self, features):
        """Predict value from a latent state."""
        # Apply hidden layers
        x = features
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Predict value
        value = self.value_layer(x)
        return value[:, 0]


class DreamerV2:
    """DreamerV2 agent."""
    
    def __init__(
        self, 
        observation_space,
        action_space,
        device='cpu',
        use_mixed_precision=False,
        embedding_size=1024,
        recurrent_state_size=1024,
        discrete_size=32,
        discrete_classes=32,
        hidden_size=1024,
        actor_hidden_size=400,
        actor_layers=4,
        critic_hidden_size=400,
        critic_layers=4,
        decoder_hidden_size=400,
        decoder_layers=2,
        sequence_length=50,
        # imagination_horizon=15,
        imagination_horizon=25,
        gamma=0.99,
        lambda_=0.95,
        # actor_entropy=1e-4,
        training_interval = 5, # Number of steps between training updates
        actor_entropy=1e-3,
        actor_grad='dynamics',  # 'reinforce' or 'dynamics'
        actor_lr=1e-4,
        critic_lr=1e-4,
        model_lr=3e-4,
        kl_weight=1.0,
        kl_balance=0.8,  # Between 0 and 1, 0.5 means equal weight posterior vs prior
        free_nats=1.0,  # Minimum KL divergence
        model_gradient_clip=100.0,
        actor_gradient_clip=100.0,
        critic_gradient_clip=100.0,
        training_steps=1,
        checkpoint_dir='./checkpoints',
        load_checkpoint=False
    ):
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        self.use_mixed_precision = use_mixed_precision
        self.embedding_size = embedding_size
        self.recurrent_state_size = recurrent_state_size
        self.discrete_size = discrete_size
        self.discrete_classes = discrete_classes
        self.hidden_size = hidden_size
        self.actor_hidden_size = actor_hidden_size
        self.actor_num_layers = actor_layers
        self.critic_hidden_size = critic_hidden_size
        self.critic_num_layers = critic_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_layers = decoder_layers
        self.sequence_length = sequence_length
        self.imagination_horizon = imagination_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.actor_entropy = actor_entropy
        self.actor_grad = actor_grad
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.model_lr = model_lr
        self.kl_weight = kl_weight
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        self.model_gradient_clip = model_gradient_clip
        self.actor_gradient_clip = actor_gradient_clip
        self.critic_gradient_clip = critic_gradient_clip
        self.training_steps = training_steps
        self.checkpoint_dir = checkpoint_dir
        self.load_checkpoint = load_checkpoint
        
        # Determine if actions are discrete or continuous
        self.discrete_actions = hasattr(self.action_space, 'n')
        
        if self.discrete_actions:
            self.action_size = self.action_space.n
        else:
            self.action_size = self.action_space.shape[0]
        
        # Set observation shape
        self.observation_shape = self.observation_space.shape


        # Set up mixed precision if requested
        if self.use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Using mixed precision training with policy:", policy.name)

        # Create networks
        self.build_networks()
        
        # Create optimizers
        if self.use_mixed_precision:
            # Use loss scaling with mixed precision to prevent underflow
            self.model_optimizer = tf.keras.optimizers.legacy.Adam(self.model_lr)
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(self.actor_lr)
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(self.critic_lr)
        else:
            self.model_optimizer = tf.keras.optimizers.legacy.Adam(self.model_lr)
            self.actor_optimizer = tf.keras.optimizers.legacy.Adam(self.actor_lr)
            self.critic_optimizer = tf.keras.optimizers.legacy.Adam(self.critic_lr)
        
    
        # # Create optimizers
        # self.model_optimizer = tf.keras.optimizers.legacy.Adam(self.model_lr)
        # self.actor_optimizer = tf.keras.optimizers.legacy.Adam(self.actor_lr)
        # self.critic_optimizer = tf.keras.optimizers.legacy.Adam(self.critic_lr)
        
        
        # Add training step counter
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

        # Recurrent state for inference
        self.recurrent_state = None
        self.discrete_state = None

        # For tracking training metrics
        self.metrics = {}

        # Checkpoint manager
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(
            encoder=self.encoder,
            rssm=self.rssm,
            decoder=self.decoder,
            reward_predictor=self.reward_predictor,
            actor=self.actor,
            critic=self.critic,
            model_optimizer=self.model_optimizer,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            global_step=self.global_step,
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=10)

            
        # Load checkpoint if requested
        if self.load_checkpoint:
            self.load()

        if hasattr(self, 'global_step'):
            if not tf.train.latest_checkpoint(self.checkpoint_dir):
                # If fresh training, initialize to 0
                self.global_step.assign(0)
            else:
                checkpoint_name = tf.train.latest_checkpoint(self.checkpoint_dir)
                try:
                    step = int(checkpoint_name.split('-')[-1])
                    self.global_step.assign(step)
                    print(f"Restored global_step to {step} from checkpoint name")
                except:
                    print("Could not infer global_step from checkpoint name")


        # Add training step counter for TensorBoard
        self.train_step = 0
        
        # Add TensorBoard writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.checkpoint_dir, 'tensorboard', current_time)
        )

    
    def log_metrics(self, metrics, step=None):
        """Enhanced logging with component-level metrics."""
        step = step or self.train_step
        
        with self.summary_writer.as_default():
            # Log model metrics
            for key, value in metrics.items():
                if isinstance(value, tf.Tensor):
                    value = value.numpy()
                tf.summary.scalar(key, value, step=step)
                
            # World model confidence metrics
            if hasattr(self, 'current_obs') and self.current_obs is not None:
                # Compute model prediction error on current observation
                embed = self.encoder(self.current_obs[None])
                
                # Predicted vs actual observation
                if hasattr(self, 'recurrent_state') and hasattr(self, 'discrete_state'):
                    if self.recurrent_state is not None and self.discrete_state is not None:
                        # Create model features
                        model_features = tf.concat([
                            self.recurrent_state,
                            tf.reshape(self.discrete_state, [1, -1])
                        ], axis=-1)
                        
                        # Compute reconstruction loss
                        obs_dist = self.decoder(model_features)
                        prediction_error = -obs_dist.log_prob(self.current_obs[None]).numpy()
                        
                        # Log as model confidence (inverse of error)
                        tf.summary.scalar('world_model_confidence', 1.0 / (1.0 + prediction_error), step=step)


    def log_decision_attribution(self, obs, action):
        """
        Return a dictionary with:
        action_taken, action_probability, world_model_score,
        exploration_bonus, value_estimate
        """
        try:
            # Ensure action is an integer
            if hasattr(action, "numpy"):
                action = action.numpy()
            if isinstance(action, np.ndarray):
                if action.size == 1:
                    action = action.item()
                else:
                    action = int(np.argmax(action))
            action = int(action)
            
            # Process observation
            obs_t = tf.convert_to_tensor(obs[None], tf.float32)
            embed = self.encoder(obs_t)
            wm_score = float(tf.reduce_mean(tf.abs(embed)))
            
            # Handle state initialization
            if self.recurrent_state is None:
                rec = tf.zeros([1, self.recurrent_state_size])
            else:
                rec = self.recurrent_state
            
            if self.discrete_state is None:
                disc = tf.zeros([1, self.discrete_size, self.discrete_classes])
                disc = tf.reshape(disc, [1, -1])
            else:
                disc = tf.reshape(self.discrete_state, [1, -1])
            
            disc = tf.cast(disc, tf.float32)
            model_feat = tf.concat([rec, disc], axis=-1)
            
            # Get action distribution
            act_dist = self.actor(model_feat)
            
            # Calculate action probability
            if self.discrete_actions:
                probs = tf.nn.softmax(act_dist.logits)[0].numpy()
                action_prob = float(probs[action])
                exploration_bonus = float(
                    tfp.distributions.Categorical(logits=act_dist.logits).entropy().numpy()
                )
            else:
                action_prob = 0.0
                exploration_bonus = float(act_dist.entropy().numpy())
            
            value_output = self.critic(model_feat)
            # Check dimensionality and handle appropriately
            if len(value_output.shape) > 1:
                value_est = float(value_output[0, 0].numpy())
            else:
                # Handle as 1D tensor
                value_est = float(value_output[0].numpy() if value_output.shape[0] > 0 else 0.0)
            
            return {
                "action_taken": action,
                "action_probability": action_prob,
                "world_model_score": wm_score,
                "exploration_bonus": exploration_bonus,
                "value_estimate": value_est,
            }
        except Exception as e:
            print(f"Unexpected error in log_decision_attribution: {e}")
            return {
                "action_taken": action if isinstance(action, int) else 0,
                "action_probability": 0.0,
                "world_model_score": 0.0,
                "exploration_bonus": 0.0,
                "value_estimate": 0.0,
            }


    def build_networks(self):
        """Build every neural sub-network and warm-up the critic
        so its first ever call fixes the input size to 2048 dims.
        """
        self.encoder = Encoder(embedding_size=self.embedding_size)

        self.rssm = RSSM(
            embedding_size=self.embedding_size,
            recurrent_state_size=self.recurrent_state_size,
            discrete_size=self.discrete_size,
            discrete_classes=self.discrete_classes,
            hidden_size=self.hidden_size,
        )

        self.decoder = Decoder(output_shape=self.observation_shape)

        self.reward_predictor = DenseDecoder(
            output_shape=[1],
            hidden_size=self.decoder_hidden_size,
            num_layers=self.decoder_num_layers,
            dist="normal",
        )

        # 3. policy & value networks 
        feat_dim = self.recurrent_state_size + self.discrete_size * self.discrete_classes  # 1024 + 1024 = 2048

        self.actor  = Actor(
            action_size=self.action_size,
            hidden_size=self.actor_hidden_size,
            num_layers=self.actor_num_layers,
            discrete=self.discrete_actions,
        )

        self.critic = Critic(
            hidden_size=self.critic_hidden_size,
            num_layers=self.critic_num_layers,
        )

        # one time warm-up so the critic is born 2048-wide 
        dummy_feat = tf.zeros([1, feat_dim], dtype=tf.float32)  # shape (1, 2048)
        _ = self.critic(dummy_feat)     # establishes weight matrix [2048, …]


    def init_state(self, batch_size=1):
        """Initialize the recurrent state."""
        recurrent_state, discrete_state = self.rssm.initial_state(batch_size)
        self.recurrent_state = recurrent_state
        self.discrete_state = discrete_state
        return recurrent_state, discrete_state


    @tf.function
    def policy(self, obs, recurrent_state=None, discrete_state=None, training=False, explore=True):
        """Generate action from observation."""
        # Ensure observation has correct shape
        if len(obs.shape) == 3:  # Single observation
            obs = obs[None, :]  # Add batch dimension
        
        # Encode observation
        embed = self.encoder(obs)
        
        # Use passed states or initialize if None
        if recurrent_state is None or discrete_state is None:
            recurrent_state, discrete_state = self.rssm.initial_state(obs.shape[0])
        
        # Update state with observation
        dummy_action = tf.zeros([obs.shape[0], self.action_size], dtype=tf.float32)
        recurrent_state, discrete_state, _, _ = self.rssm.observe(
            embed, dummy_action, recurrent_state, discrete_state)
        
        # Create model state
        model_state = tf.concat([
            recurrent_state,
            tf.cast(tf.reshape(discrete_state, [discrete_state.shape[0], -1]), tf.float32)
        ], axis=-1)
        
        # Get action distribution
        action_dist = self.actor(model_state)
        
        # Sample action
        if explore:
            action = action_dist.sample()
        else:
            if self.discrete_actions:
                action = tf.one_hot(tf.argmax(action_dist.logits, axis=-1), self.action_size)
            else:
                action = action_dist.mean()
        
        # Process discrete actions
        if self.discrete_actions:
            action = tf.argmax(action, axis=-1)
        
        # Return action AND the updated states
        return action, recurrent_state, discrete_state


    @tf.function(experimental_relax_shapes=True)
    def train_batch(self, obs, actions, rewards, dones, next_obs):
        """Train on a batch of data."""
        # Ensure inputs have correct shape and type - only normalize observations
        obs = tf.cast(tf.convert_to_tensor(obs), tf.float32) / 255.0
        next_obs = tf.cast(tf.convert_to_tensor(next_obs), tf.float32) / 255.0
        
        # Convert other inputs without normalization
        if self.discrete_actions:
            actions = tf.one_hot(tf.cast(actions, tf.int32), self.action_size)
        else:
            actions = tf.cast(tf.convert_to_tensor(actions), tf.float32)
        
        rewards = tf.cast(tf.convert_to_tensor(rewards), tf.float32)
        dones = tf.cast(tf.convert_to_tensor(dones), tf.float32)
        
        # Store metrics
        metrics = {}
        
        # Train model
        model_loss, model_metrics = self.train_model(obs, actions, rewards, dones, next_obs)
        
        for k, v in model_metrics.items():
            metrics[f'model/{k}'] = v
        metrics['model/loss'] = model_loss
        
        # Train actor and critic
        actor_loss, critic_loss, actor_metrics, critic_metrics = self.train_actor_critic(
            obs, actions, rewards, dones)
        
        for k, v in actor_metrics.items():
            metrics[f'actor/{k}'] = v
        for k, v in critic_metrics.items():
            metrics[f'critic/{k}'] = v
        metrics['actor/loss'] = actor_loss
        metrics['critic/loss'] = critic_loss
        
        self.train_step += 1
        self.global_step.assign_add(1)

        return metrics

    @tf.function(experimental_relax_shapes=True)
    def train_model(self, obs, actions, rewards, dones, next_obs):
        """Train the world model."""
        batch_size = obs.shape[0]
        sequence_length = obs.shape[1]
        metrics = {}
        
        with tf.GradientTape() as tape:
            # Reshape to merge batch and sequence dimensions
            obs_shape = obs.shape
            obs_reshaped = tf.reshape(obs, [-1] + list(self.observation_shape))
            
            # Encode all observations at once
            all_embeds = self.encoder(obs_reshaped)
            
            # Reshape back to include sequence dimension
            embeds = tf.reshape(all_embeds, [batch_size, sequence_length, self.embedding_size])
            
            # Initialize states
            recurrent_state, discrete_state = self.rssm.initial_state(batch_size)
            
            # Variables to collect
            all_recurrent_states = []
            all_discrete_states = []
            all_prior_dists = []
            all_posterior_dists = []
            
            # Process sequence
            for t in range(sequence_length - 1):  # -1 because we need next_obs
                # Current action and next embed
                action = actions[:, t]
                next_embed = embeds[:, t + 1]
                
                # Update states
                recurrent_state, discrete_state, prior_dist, posterior_dist = self.rssm.observe(
                    next_embed, action, recurrent_state, discrete_state)
                
                # Collect variables
                all_recurrent_states.append(recurrent_state)
                all_discrete_states.append(discrete_state)
                all_prior_dists.append(prior_dist)
                all_posterior_dists.append(posterior_dist)
            
            # Stack collected variables
            all_recurrent_states = tf.stack(all_recurrent_states, axis=1)
            all_discrete_states = tf.stack(all_discrete_states, axis=1)
            
            # Create model features
            flat_recurrent_states = tf.reshape(all_recurrent_states, [-1, self.recurrent_state_size])
            flat_discrete_states = tf.reshape(all_discrete_states, 
                                           [-1, self.discrete_size * self.discrete_classes])
            model_features = tf.concat([
                            flat_recurrent_states, 
                            tf.cast(flat_discrete_states, tf.float32)
                        ], axis=-1)

            
            # Predict reconstructions
            flat_obs = tf.reshape(obs[:, 1:], [-1] + list(self.observation_shape))
            obs_dist = self.decoder(model_features)
            obs_loss = -tf.reduce_mean(obs_dist.log_prob(flat_obs))
            metrics['obs_loss'] = obs_loss
            
            # Predict rewards
            flat_rewards = tf.reshape(rewards[:, :-1], [-1, 1])
            reward_dist = self.reward_predictor(model_features)
            reward_loss = -tf.reduce_mean(reward_dist.log_prob(flat_rewards))
            metrics['reward_loss'] = reward_loss
            
            # Calculate KL loss with improved balancing approach
            kl_loss = 0
            for t in range(len(all_prior_dists)):
                prior_dist = all_prior_dists[t]
                posterior_dist = all_posterior_dists[t]
                
                # Calculate KL divergence with improved balancing
                if self.kl_balance == 0.5:
                    kl_divergence = tfd.kl_divergence(posterior_dist, prior_dist)
                else:
                    # Enhanced KL balancing from original DreamerV2 implementation
                    alpha = self.kl_balance
                    
                    # KL from posterior to prior (standard KL term)
                    kl_prior = tfd.kl_divergence(posterior_dist, prior_dist)
                    
                    # KL from posterior to uniform distribution (entropy term)
                    # Create uniform distribution with same shape as prior
                    uniform_logits = tf.ones_like(prior_dist.distribution.logits) / self.discrete_classes
                    uniform_dist = tfd.Independent(tfd.OneHotCategorical(probs=uniform_logits), 1)
                    kl_post = tfd.kl_divergence(posterior_dist, uniform_dist)
                    
                    # Balanced KL divergence
                    kl_divergence = alpha * kl_prior + (1 - alpha) * kl_post
                
                # Apply free bits - minimum KL to ensure some information flow
                kl_divergence = tf.maximum(kl_divergence, self.free_nats)
                
                # Reduce per batch element and accumulate
                kl_loss += tf.reduce_mean(kl_divergence)
                
                # Log individual KL components for better analysis
                if t == 0:  # Only log once for efficiency
                    tf.summary.scalar('kl/prior_component', tf.reduce_mean(kl_prior), step=self.train_step)
                    tf.summary.scalar('kl/uniform_component', tf.reduce_mean(kl_post), step=self.train_step)
                    tf.summary.scalar('kl/balanced', tf.reduce_mean(kl_divergence), step=self.train_step)

            
            # Average over time
            kl_loss /= len(all_prior_dists)
            metrics['kl_loss'] = kl_loss
            
            # Combine losses
            model_loss = obs_loss + reward_loss + self.kl_weight * kl_loss
            
        # Compute gradients and update model
        model_vars = (
            self.encoder.trainable_variables + 
            self.rssm.trainable_variables + 
            self.decoder.trainable_variables + 
            self.reward_predictor.trainable_variables
        )
        
        model_grads = tape.gradient(model_loss, model_vars)
        
        # Clip gradients
        model_grads, _ = tf.clip_by_global_norm(model_grads, self.model_gradient_clip)
        
        # Apply gradients
        self.model_optimizer.apply_gradients(zip(model_grads, model_vars))
        
        return model_loss, metrics
    

    @tf.function(experimental_relax_shapes=True)
    def train_actor_critic(self, obs, actions, rewards, dones):
        """Train the actor and critic."""
        batch_size = obs.shape[0]
        sequence_length = obs.shape[1]
        metrics = {}
        
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            # Reshape to merge batch and sequence dimensions
            obs_reshaped = tf.reshape(obs, [-1] + list(self.observation_shape))
            
            # Encode all observations at once
            all_embeds = self.encoder(obs_reshaped)
            
            # Reshape back to include sequence dimension
            embeds = tf.reshape(all_embeds, [batch_size, sequence_length, self.embedding_size])
            
            
            # Initialize states
            recurrent_state, discrete_state = self.rssm.initial_state(batch_size)
            
            # Variables to collect
            all_recurrent_states = []
            all_discrete_states = []
            
            # Process sequence
            for t in range(obs.shape[1] - 1):  # -1 because we need next_obs
                # Current action and next embed
                action = actions[:, t]
                next_embed = embeds[:, t + 1]
                
                # Update states
                recurrent_state, discrete_state, _, _ = self.rssm.observe(
                    next_embed, action, recurrent_state, discrete_state)
                
                # Collect variables
                all_recurrent_states.append(recurrent_state)
                all_discrete_states.append(discrete_state)
            
            # Stack collected variables
            all_recurrent_states = tf.stack(all_recurrent_states, axis=1)
            all_discrete_states = tf.stack(all_discrete_states, axis=1)
            
            # Sample starting states for imagination
            sequence_indices = tf.random.uniform(
                [batch_size], minval=0, maxval=all_recurrent_states.shape[1], dtype=tf.int32)
            
            imag_recurrent_states = []
            imag_discrete_states = []
            imag_actions = []
            imag_rewards = []
            imag_values = []
            
            # Get initial states
            init_recurrent_state = tf.gather_nd(
                all_recurrent_states, 
                tf.stack([tf.range(batch_size), sequence_indices], axis=1))
            init_discrete_state = tf.gather_nd(
                all_discrete_states, 
                tf.stack([tf.range(batch_size), sequence_indices], axis=1))
            
            # Start with real states
            recurrent_state = init_recurrent_state
            discrete_state = init_discrete_state
            
            # Imagine trajectories
            for t in range(self.imagination_horizon):
                # Create model features
                model_features = tf.concat([
                                recurrent_state, 
                                tf.cast(tf.reshape(discrete_state, [batch_size, -1]), tf.float32)
                            ], axis=-1)

                
                # Predict action
                action_dist = self.actor(model_features)
                action = action_dist.sample()
                
                # Calculate entropy for exploration
                if self.discrete_actions:
                    entropies = -tf.reduce_sum(
                        tf.nn.softmax(action_dist.logits) * 
                        tf.nn.log_softmax(action_dist.logits), 
                        axis=-1)

                else:
                    entropies = action_dist.entropy()
                
                # Imagine next state
                recurrent_state, discrete_state, _ = self.rssm.imagine(
                    action, recurrent_state, discrete_state)
                
                # Create new model features
                model_features = tf.concat([
                                    recurrent_state,
                                    tf.cast(tf.reshape(discrete_state, [batch_size, -1]), 
                                    tf.float32)], axis=-1)

                
                # Predict reward
                reward_dist = self.reward_predictor(model_features)
                reward = reward_dist.mean()
                
                # Predict value
                value = self.critic(model_features)
                
                # Collect imagined trajectory
                imag_recurrent_states.append(recurrent_state)
                imag_discrete_states.append(discrete_state)
                imag_actions.append(action)
                imag_rewards.append(reward)
                imag_values.append(value)
            
            # Stack imagined trajectory
            imag_recurrent_states = tf.stack(imag_recurrent_states, axis=1)
            imag_discrete_states = tf.stack(imag_discrete_states, axis=1)
            imag_actions = tf.stack(imag_actions, axis=1)
            imag_rewards = tf.stack(imag_rewards, axis=1)
            imag_values = tf.stack(imag_values, axis=1)
            
            # Compute lambda returns
            returns = self.compute_return(imag_rewards, imag_values, self.gamma, self.lambda_)
            
            # Model features for actor and critic loss
            flat_recurrent_states = tf.reshape(imag_recurrent_states, [-1, self.recurrent_state_size])
            flat_discrete_states = tf.reshape(imag_discrete_states, 
                                             [-1, self.discrete_size * self.discrete_classes])
            flat_model_features = tf.concat([flat_recurrent_states, tf.cast(flat_discrete_states, tf.float32)], axis=-1)

            
            # Compute action entropy for all steps
            flat_actions = tf.reshape(imag_actions, [-1, self.action_size])
            actor_dists = self.actor(flat_model_features)
            if self.discrete_actions:
                entropies = -tf.reduce_sum(
                    tf.nn.softmax(actor_dists.logits) * 
                    tf.nn.log_softmax(actor_dists.logits), 
                    axis=-1)
            else:
                entropies = actor_dists.entropy()

            # Reshape entropy to match returns
            entropies = tf.reshape(entropies, [batch_size, self.imagination_horizon])

                        
            # Compute critic targets and loss
            flat_values = self.critic(flat_model_features)
            values = tf.reshape(flat_values, [batch_size, self.imagination_horizon])
            
            # Shape returns to match values
            returns = tf.stop_gradient(returns)
            critic_loss = tf.reduce_mean(0.5 * tf.square(returns - values))
            metrics['returns'] = tf.reduce_mean(returns)
            metrics['values'] = tf.reduce_mean(values)
            
            # Compute actor loss
            if self.actor_grad == 'dynamics':
                # Use dynamics backpropagation (Dreamer)
                actor_loss = -tf.reduce_mean(returns)
            else:
                # Use REINFORCE gradient (DreamerV2)
                advantage = tf.stop_gradient(returns - values)
                actor_loss = -tf.reduce_mean(entropies * advantage)
            
            # Add entropy regularization
            actor_loss -= self.actor_entropy * tf.reduce_mean(entropies)
            
            # Track entropy
            metrics['entropy'] = tf.reduce_mean(entropies)
        
        # Get actor variables
        actor_vars = self.actor.trainable_variables
        
        # Get critic variables
        critic_vars = self.critic.trainable_variables
        
        # Compute actor gradients
        actor_grads = actor_tape.gradient(actor_loss, actor_vars)
        
        # Compute critic gradients
        critic_grads = critic_tape.gradient(critic_loss, critic_vars)
        
        # Clip gradients
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.actor_gradient_clip)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.critic_gradient_clip)
        
        # Apply gradients
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))
        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))
        
        return actor_loss, critic_loss, metrics, {}
    
    def compute_return(self, rewards, values, gamma, lambda_):
        """Compute lambda returns."""

        # Squeeze rewards to match the shape of values
        rewards = tf.squeeze(rewards, axis=-1)    

        # Compute GAE (Generalized Advantage Estimation)
        next_values = tf.concat([values[:, 1:], tf.zeros_like(values[:, :1])], axis=1)
        delta = rewards + gamma * next_values - values
        advantage = delta
        
        # Compute lambda return using GAE
        returns = []
        for t in range(delta.shape[1] - 1, -1, -1):
            if t == delta.shape[1] - 1:
                returns.append(delta[:, t])
            else:
                returns.append(delta[:, t] + gamma * lambda_ * advantage)
            advantage = returns[-1]
        
        # Reverse the list and stack
        returns = tf.stack(returns[::-1], axis=1)
        
        # Add value to get full returns
        returns = returns + values
        
        return returns
    
    def save(self, step=None):
        """Save model checkpoint."""
        if step is None:
            step = self.global_step.numpy()
        save_path = self.manager.save(checkpoint_number=step)
        print(f"Saved checkpoint: {save_path}")

        
    def load(self):
        """Load checkpoint, skipping anything that no longer fits."""
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        if not latest:
            print("No checkpoint found – starting from scratch.")
            return False

        print("Restoring from", latest)
        self.checkpoint.restore(latest).expect_partial()
        return True


