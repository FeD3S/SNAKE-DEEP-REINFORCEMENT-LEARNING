import environments_fully_observable 
import environments_partially_observable
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import os # Imported os to handle directory creation for saving weights

# ==============================================================================
# ENVIRONMENT PATCH
# ==============================================================================
def partial_to_state(self):
    """
    Generates an ego-centric state representation for partially observable environments.
    Extracts a local grid centered on the snake's head.
    """
    # Locate the snake's head on every parallel board (returns [board_idx, x, y])
    heads = np.argwhere(self.boards == self.HEAD)
    
    # Pad the outer edges of the boards with WALL values to prevent out-of-bounds 
    # errors when the snake is near the edge.
    padded_boards = np.pad(self.boards, 
                           pad_width=((0, 0), (self.mask_size, self.mask_size), (self.mask_size, self.mask_size)), 
                           mode='constant', 
                           constant_values=self.WALL)
                           
    # Pre-allocate an empty tensor to hold the extracted local views.
    # Dimensions: (batch_size, 2*radius + 1, 2*radius + 1)
    states = np.zeros((self.n_boards, 2 * self.mask_size + 1, 2 * self.mask_size + 1))
    
    # Extract the local view for each board in the batch.
    for i in range(self.n_boards):
        # Shift the raw head coordinates to account for the padding
        hx = heads[i, 1] + self.mask_size
        hy = heads[i, 2] + self.mask_size
        
        # Slice the padded board to get the exact square window centered on the head
        states[i] = padded_boards[i, 
                                  hx - self.mask_size : hx + self.mask_size + 1, 
                                  hy - self.mask_size : hy + self.mask_size + 1]
                                  
    # Convert integer grids into one-hot encoded tensors (classes 0-4).
    # [..., 1:] drops the 0th channel (WALL), implicitly representing walls as all zeros.
    return tf.keras.utils.to_categorical(states, num_classes=5)[..., 1:]

# Override the original method with the patched version
environments_partially_observable.OriginalSnakeEnvironment.to_state = partial_to_state

# ==============================================================================
# INITIALIZATION & HYPERPARAMETERS
# ==============================================================================
# Set seeds for reproducibility
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

def get_envs(n=1000, size=10, mask_size=2):
    env_full = environments_fully_observable.OriginalSnakeEnvironment(n, size)
    env_part = environments_partially_observable.OriginalSnakeEnvironment(n, size, mask_size)
    return env_full, env_part

env_full, env_part = get_envs()

# RL Hyperparameters
GAMMA = 0.975          # Discount factor for future rewards
ITERATIONS_full = 80000   # Total training steps
ITERATIONS_part = 80000   # Total training steps
SPEED = 0.0001        # Pause duration for live plotting (smaller -> faster)

# ==============================================================================
# NEURAL NETWORK ARCHITECTURES
# ==============================================================================
def build_global_mlp_network(input_shape):
    """Builds a Multi-Layer Perceptron (MLP) for Q-value approximation."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
   
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    # 4 outputs corresponding to Q-values for Up, Down, Left, Right
    q_values = tf.keras.layers.Dense(4)(x)
    return tf.keras.Model(inputs=inputs, outputs=q_values)

# ==============================================================================
# NETWORK INSTANTIATION & OPTIMIZERS
# ==============================================================================
# Fully observable networks (Input shape: 10x10 map * 4 channels)
q_full = build_global_mlp_network((10, 10, 4))
target_q_full = build_global_mlp_network((10, 10, 4))
target_q_full.set_weights(q_full.get_weights()) # Sync initial weights

# Partially observable networks (Input shape: 5x5 map * 4 channels)
q_part = build_global_mlp_network((5, 5, 4))
target_q_part = build_global_mlp_network((5, 5, 4))
target_q_part.set_weights(q_part.get_weights()) # Sync initial weights

lr_schedule_part = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5.5e-4,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

# Optimizers
optimizer_full = tf.keras.optimizers.Adam(5e-4)
optimizer_part = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule_part
)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def get_action_mask(env):
    """
    Generates a binary mask (1.0 for valid, 0.0 for invalid) to prevent the agent 
    from intentionally moving into walls or its own body.
    """
    heads = np.argwhere(env.boards == env.HEAD)
    mask = np.ones((env.n_boards, 4), dtype=np.float32)
    
    # Direction vectors: [Down, Right, Up, Left] (depending on environment axes)
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    
    for a in range(4):
        # Calculate next potential position
        new_x = heads[:, 1] + dx[a]
        new_y = heads[:, 2] + dy[a]
        
        # Clip to prevent index out-of-bounds
        valid_x = np.clip(new_x, 0, env.board_size - 1)
        valid_y = np.clip(new_y, 0, env.board_size - 1)
        
        # Mask out wall collisions
        is_wall = env.boards[heads[:, 0], valid_x, valid_y] == env.WALL
        mask[is_wall, a] = 0.0 
        
        # Mask out body collisions
        is_body = env.boards[heads[:, 0], valid_x, valid_y] == env.BODY
        mask[is_body, a] = 0.0 
        
    # Fallback: If all 4 moves are masked (certain death), unmask all to prevent 
    # network sampling crashes (e.g., categorical distribution errors).
    all_masked = tf.reduce_sum(mask, axis=1) == 0
    mask_fallback = tf.where(tf.expand_dims(all_masked, 1), tf.ones_like(mask), mask)
        
    return tf.convert_to_tensor(mask_fallback)

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train_dqn(env, q_network, target_q_network, optimizer, iterations, epsilon_decay, epsilon_min, title="Training", live_plot_freq=1):
    """Trains the Deep Q-Network using a Double DQN approach with action masking."""
    rewards_history = []
    food_history = []
    length_history = []
    
    # Epsilon-greedy exploration parameters
    epsilon = 1.0           
    update_target_every = 100 
    
    # Setup live plotting
    plt.ion() 
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    if hasattr(fig.canvas, 'manager'):
        fig.canvas.manager.set_window_title(title)
    
    for iteration in trange(iterations):
        state = env.to_state()
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        with tf.GradientTape() as tape:
            # 1. Forward pass online network
            q_vals = q_network(state)
            mask = get_action_mask(env)
            
            # Apply large negative penalty to invalid actions
            masked_q_vals = q_vals + (mask - 1.0) * 1e9
            
            # 2. Select actions (Epsilon-Greedy)
            if np.random.rand() < epsilon:
                # Random action (respecting the mask)
                noise = tf.random.uniform(q_vals.shape)
                masked_noise = noise + (mask - 1.0) * 1e9
                actions = tf.argmax(masked_noise, axis=-1, output_type=tf.int32)
            else:
                # Best action according to Q-network
                actions = tf.argmax(masked_q_vals, axis=-1, output_type=tf.int32)
                
            # 3. Step the environment
            rewards = env.move(actions)
            new_state = tf.constant(env.to_state(), dtype=tf.float32)
            
            # --- Metrics Tracking ---
            food_eaten_this_step = tf.reduce_sum(tf.cast(rewards > 0, tf.float32)).numpy()
            food_history.append(food_eaten_this_step / env.n_boards)
            
            current_lengths = [len(b) + 1 for b in env.bodies]
            length_history.append(np.mean(current_lengths))
            
            # --- Double DQN Target Calculation ---
            # Use online network to select next best action
            next_q_vals_online = q_network(new_state)
            next_mask = get_action_mask(env)
            masked_next_q_vals_online = next_q_vals_online + (next_mask - 1.0) * 1e9
            next_actions = tf.argmax(masked_next_q_vals_online, axis=-1, output_type=tf.int32)
            
            # Use target network to evaluate that selected action
            next_q_vals_target = target_q_network(new_state)
            next_action_masks = tf.one_hot(next_actions, 4)
            max_next_q = tf.reduce_sum(tf.multiply(next_q_vals_target, next_action_masks), axis=-1)
            
            # Calculate Bellman target
            target_q = tf.squeeze(rewards) + GAMMA * max_next_q
            
            # Calculate predicted Q-values for actions actually taken
            action_masks = tf.one_hot(actions, 4)
            predicted_q = tf.reduce_sum(tf.multiply(q_vals, action_masks), axis=-1)
            
            # Compute Huber loss (robust to outliers compared to MSE)
            loss = tf.keras.losses.Huber()(target_q, predicted_q)
            
        # 4. Backpropagation
        gradients = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
        
        # 5. Periodically update target network
        if iteration % update_target_every == 0:
            target_q_network.set_weights(q_network.get_weights())
        
        rewards_history.append(np.mean(rewards))

        # --- Live Plotting ---
        if iteration % live_plot_freq == 0:
            # Plot game state of board 0
            ax1.clear()
            ax1.imshow(env.boards[0], origin="lower")
            ax1.set_title(f"Board 0 - Iter {iteration}")
            ax1.axis('off')

            # Plot average reward (smoothed)
            ax2.clear()
            window = 200
            if len(rewards_history) > window:
                smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
                ax2.plot(smoothed)
            else:
                ax2.plot(rewards_history)
            ax2.set_title("Average Reward (Smoothed)")
            ax2.grid(True)
            
            # Plot average food eaten
            ax3.clear()
            window_m = 100
            if len(food_history) > window_m:
                smoothed_f = np.convolve(food_history, np.ones(window_m), mode='valid')
                ax3.plot(smoothed_f, color='green')
            else:
                cum_food = [sum(food_history[:i+1]) for i in range(len(food_history))]
                ax3.plot(cum_food, color='green')
            ax3.set_title("Avg Food / 100 Iters")
            ax3.grid(True)
            
            # Plot average snake length
            ax4.clear()
            if len(length_history) > window_m:
                smoothed_l = np.convolve(length_history, np.ones(window_m)/window_m, mode='valid')
                ax4.plot(smoothed_l, color='blue')
            else:
                ax4.plot(length_history, color='blue')
            ax4.set_title("Avg Length / 100 Iters")
            ax4.grid(True)
            
            plt.draw()
            plt.pause(SPEED) 
            
    plt.ioff()
    plt.close(fig)
    return rewards_history, food_history, length_history

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":

    # Create a 'weights' folder in the current working directory.
    os.makedirs('weights', exist_ok=True)

    # User selection for training routines
    choice = input("\nDo you want to train full, partial or both? [1, 2, 12]: ").strip().lower()
    
    # Initialize variables to keep track of what was trained to avoid undefined errors later
    trained_full = False
    trained_part = False

    if choice == '1':     
        print("Training Fully Observable Environment...")
        history_full, food_full, len_full = train_dqn(env_full, q_full, target_q_full, optimizer_full, ITERATIONS_full, epsilon_decay = 0.99992, epsilon_min = 0.0001,  title="Fully Observable Training")
        trained_full = True
        
        # Save the fully observable model weights in HDF5 format
        q_full.save_weights('weights/q_full.weights.h5')
        print("Successfully saved weights for fully observable model to 'weights/q_full.weights.h5'")

    elif choice == '2':
        print("Training Partially Observable Environment...")
        history_part, food_part, len_part = train_dqn(env_part, q_part, target_q_part, optimizer_part, ITERATIONS_part, epsilon_decay = 0.99992, epsilon_min = 0.1,  title="Partially Observable Training")
        trained_part = True
        
        # Save the partially observable model weights in HDF5 format
        q_part.save_weights('weights/q_part.weights.h5')
        print("Successfully saved weights for partially observable model to 'weights/q_part.weights.h5'")
        
    elif choice == '12':
        print("Training Both Environments...")
        history_full, food_full, len_full = train_dqn(env_full, q_full, target_q_full, optimizer_full, ITERATIONS_full, epsilon_decay = 0.99992, epsilon_min = 0.0001,  title="Fully Observable Training")
        trained_full = True
        # Save fully observable model
        q_full.save_weights('weights/q_full.weights.h5')
        print("Successfully saved weights for fully observable model to 'weights/q_full.weights.h5'")
        
        history_part, food_part, len_part = train_dqn(env_part, q_part, target_q_part, optimizer_part, ITERATIONS_part, epsilon_decay = 0.99992, epsilon_min = 0.1,    title="Partially Observable Training")
        trained_part = True
        # Save partially observable model
        q_part.save_weights('weights/q_part.weights.h5')
        print("Successfully saved weights for partially observable model to 'weights/q_part.weights.h5'")

    # ==============================================================================
    # FINAL VISUALIZATION
    # ==============================================================================
    # Define window sizes for smoothing the data to reduce noise in the plots
    window_r = 200
    window_m = 100
    
    if trained_full:
        # Smooth data for Fully Observable plots using a moving average (convolution)
        smoothed_r_full = np.convolve(history_full, np.ones(window_r)/window_r, mode='valid')
        smoothed_f_full = np.convolve(food_full, np.ones(window_m), mode='valid')
        smoothed_l_full = np.convolve(len_full, np.ones(window_m)/window_m, mode='valid')
        
        fig1 = plt.figure(figsize=(6, 5))
        if hasattr(fig1.canvas, 'manager'):
            fig1.canvas.manager.set_window_title("Fully Obs: Average Reward")
        plt.plot(smoothed_r_full, color='blue')
        plt.title('Fully Obs: Average Reward')
        plt.grid(True)
        
        fig2 = plt.figure(figsize=(6, 5))
        if hasattr(fig2.canvas, 'manager'):
            fig2.canvas.manager.set_window_title("Fully Obs: Food / 100 Iters")
        plt.plot(smoothed_f_full, color='green')
        plt.title('Fully Obs: Food / 100 Iters')
        plt.grid(True)
        
        fig3 = plt.figure(figsize=(6, 5))
        if hasattr(fig3.canvas, 'manager'):
            fig3.canvas.manager.set_window_title("Fully Obs: Average Length")
        plt.plot(smoothed_l_full, color='red')
        plt.title('Fully Obs: Average Length')
        plt.grid(True)
        
    if trained_part:
        # Smooth data for Partially Observable plots using a moving average
        smoothed_r_part = np.convolve(history_part, np.ones(window_r)/window_r, mode='valid')
        smoothed_f_part = np.convolve(food_part, np.ones(window_m), mode='valid')
        smoothed_l_part = np.convolve(len_part, np.ones(window_m)/window_m, mode='valid')

        fig4 = plt.figure(figsize=(6, 5))
        if hasattr(fig4.canvas, 'manager'):
            fig4.canvas.manager.set_window_title("Partially Obs: Average Reward")
        plt.plot(smoothed_r_part, color='blue')
        plt.title('Partially Obs: Average Reward')
        plt.grid(True)
        
        fig5 = plt.figure(figsize=(6, 5))
        if hasattr(fig5.canvas, 'manager'):
            fig5.canvas.manager.set_window_title("Partially Obs: Food / 100 Iters")
        plt.plot(smoothed_f_part, color='green')
        plt.title('Partially Obs: Food / 100 Iters')
        plt.grid(True)
        
        fig6 = plt.figure(figsize=(6, 5))
        if hasattr(fig6.canvas, 'manager'):
            fig6.canvas.manager.set_window_title("Partially Obs: Average Length")
        plt.plot(smoothed_l_part, color='red')
        plt.title('Partially Obs: Average Length')
        plt.grid(True)
        
    if trained_full and trained_part:
        fig7 = plt.figure(figsize=(6, 5))
        if hasattr(fig7.canvas, 'manager'):
            fig7.canvas.manager.set_window_title("Comparison: Average Reward")
        plt.plot(smoothed_r_full, label='Fully Obs')
        plt.plot(smoothed_r_part, label='Partially Obs')
        plt.title('Comparison: Average Reward')
        plt.legend()
        plt.grid(True)
        
        fig8 = plt.figure(figsize=(6, 5))
        if hasattr(fig8.canvas, 'manager'):
            fig8.canvas.manager.set_window_title("Comparison: Food / 100 Iters")
        plt.plot(smoothed_f_full, label='Fully Obs')
        plt.plot(smoothed_f_part, label='Partially Obs')
        plt.title('Comparison: Food / 100 Iters')
        plt.legend()
        plt.grid(True)
        
        fig9 = plt.figure(figsize=(6, 5))
        if hasattr(fig9.canvas, 'manager'):
            fig9.canvas.manager.set_window_title("Comparison: Average Length")
        plt.plot(smoothed_l_full, label='Fully Obs')
        plt.plot(smoothed_l_part, label='Partially Obs')
        plt.title('Comparison: Average Length')
        plt.legend()
        plt.grid(True)

    plt.show(block=True)