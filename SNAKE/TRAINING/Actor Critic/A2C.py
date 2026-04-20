import os
import environments_fully_observable 
import environments_partially_observable
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import random
import tensorflow as tf

# ==============================================================================
# ENVIRONMENT PATCH
# ==============================================================================
def partial_to_state(self):
    """
    Generates an ego-centric state representation for partially observable environments.
    Extracts a local grid centered on the snake's head.
    """
    heads = np.argwhere(self.boards == self.HEAD)
    
    padded_boards = np.pad(self.boards, 
                           pad_width=((0, 0), (self.mask_size, self.mask_size), (self.mask_size, self.mask_size)), 
                           mode='constant', 
                           constant_values=self.WALL)
                           
    states = np.zeros((self.n_boards, 2 * self.mask_size + 1, 2 * self.mask_size + 1))
    
    for i in range(self.n_boards):
        hx = heads[i, 1] + self.mask_size
        hy = heads[i, 2] + self.mask_size
        states[i] = padded_boards[i, 
                                  hx - self.mask_size : hx + self.mask_size + 1, 
                                  hy - self.mask_size : hy + self.mask_size + 1]
                                  
    return tf.keras.utils.to_categorical(states, num_classes=5)[..., 1:]

environments_partially_observable.OriginalSnakeEnvironment.to_state = partial_to_state

# ==============================================================================
# INITIALIZATION & HYPERPARAMETERS
# ==============================================================================
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

def get_envs(n=1000, size=10, mask_size=2):
    env_full = environments_fully_observable.OriginalSnakeEnvironment(n, size)
    env_part = environments_partially_observable.OriginalSnakeEnvironment(n, size, mask_size)
    return env_full, env_part

env_full, env_part = get_envs()

# RL Hyperparameters
GAMMA = 0.975          
ITERATIONS_full = 50000   
ITERATIONS_part = 50000   
SPEED = 0.00001        

# ==============================================================================
# NEURAL NETWORK ARCHITECTURE
# ==============================================================================
def build_actor_critic_network(input_shape):
    """Builds an Actor-Critic network."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
   
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # Actor head (policy logits)
    logits = tf.keras.layers.Dense(4)(x)
    
    # Critic head (state value)
    value = tf.keras.layers.Dense(1)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=[logits, value])

# ==============================================================================
# NETWORK INSTANTIATION & OPTIMIZERS
# ==============================================================================
ac_full = build_actor_critic_network((10, 10, 4))
ac_part = build_actor_critic_network((5, 5, 4))

lr_schedule_part = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=6e-4,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

optimizer_full = tf.keras.optimizers.Adam(5e-4)
optimizer_part = tf.keras.optimizers.Adam(learning_rate=lr_schedule_part)

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
    
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    
    for a in range(4):
        new_x = heads[:, 1] + dx[a]
        new_y = heads[:, 2] + dy[a]
        
        valid_x = np.clip(new_x, 0, env.board_size - 1)
        valid_y = np.clip(new_y, 0, env.board_size - 1)
        
        is_wall = env.boards[heads[:, 0], valid_x, valid_y] == env.WALL
        mask[is_wall, a] = 0.0 
        
        is_body = env.boards[heads[:, 0], valid_x, valid_y] == env.BODY
        mask[is_body, a] = 0.0 
        
    all_masked = tf.reduce_sum(mask, axis=1) == 0
    mask_fallback = tf.where(tf.expand_dims(all_masked, 1), tf.ones_like(mask), mask)
        
    return tf.convert_to_tensor(mask_fallback)

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train_a2c(env, model, optimizer, iterations, title="Training", live_plot_freq=1):
    """Trains the Actor-Critic Network."""
    rewards_history = []
    food_history = []
    length_history = []
    
    plt.ion() 
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    if hasattr(fig.canvas, 'manager'):
        fig.canvas.manager.set_window_title(title)
    
    for iteration in trange(iterations):
        state = env.to_state()
        state = tf.constant(state, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # 1. Forward pass
            logits, values = model(state)
            mask = get_action_mask(env)
            
            # Apply mask to logits
            masked_logits = logits + (mask - 1.0) * 1e9
            
            # 2. Select actions stochastically based on policy
            actions = tf.random.categorical(masked_logits, 1, dtype=tf.int32)
            actions_squeezed = tf.squeeze(actions, axis=-1)
                
            # 3. Step the environment
            rewards = env.move(actions_squeezed)
            new_state = tf.constant(env.to_state(), dtype=tf.float32)
            
            # --- Metrics Tracking ---
            food_eaten_this_step = tf.reduce_sum(tf.cast(rewards > 0, tf.float32)).numpy()
            food_history.append(food_eaten_this_step / env.n_boards)
            
            current_lengths = [len(b) + 1 for b in env.bodies]
            length_history.append(np.mean(current_lengths))
            
            # 4. Calculate Advantage and Loss
            _, next_values = model(new_state)
            targets = rewards + GAMMA * next_values
            advantages = targets - values
            
            critic_loss = tf.keras.losses.Huber()(targets, values)
            
            log_probs = tf.nn.log_softmax(masked_logits)
            action_masks = tf.one_hot(actions_squeezed, 4)
            selected_log_probs = tf.reduce_sum(log_probs * action_masks, axis=-1, keepdims=True)
            
            # Stop gradient on advantage to avoid pulling the critic during actor update
            actor_loss = -tf.reduce_mean(selected_log_probs * tf.stop_gradient(advantages))
            
            # Entropy bonus to encourage exploration
            probs = tf.nn.softmax(masked_logits)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=-1))
            
            total_loss = actor_loss + critic_loss - 0.01 * entropy
            
        # 5. Backpropagation
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        rewards_history.append(np.mean(rewards))

        # --- Live Plotting ---
        if iteration % live_plot_freq == 0:
            ax1.clear()
            ax1.imshow(env.boards[0], origin="lower")
            ax1.set_title(f"Board 0 - Iter {iteration}")
            ax1.axis('off')

            ax2.clear()
            window = 200
            if len(rewards_history) > window:
                smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
                ax2.plot(smoothed)
            else:
                ax2.plot(rewards_history)
            ax2.set_title("Average Reward (Smoothed)")
            ax2.grid(True)
            
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
# INFERENCE / TESTING LOOP
# ==============================================================================
def visualize_game(env, model, steps=100, title="Trained Agent"):
    """Runs a fully trained agent purely on exploitation."""
    plt.ion() 
    fig, ax = plt.subplots(figsize=(5, 5))
    if hasattr(fig.canvas, 'manager'):
        fig.canvas.manager.set_window_title(title)
    
    for step in range(steps):
        state = tf.constant(env.to_state(), dtype=tf.float32)
        logits, _ = model(state)
        mask = get_action_mask(env)
        
        masked_logits = logits + (mask - 1.0) * 1e9
        action = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)
        
        env.move(action)
        
        ax.clear()
        ax.imshow(env.boards[0], origin="lower")
        ax.set_title(f"{title} - Step {step+1}")
        ax.axis('off')
        plt.draw()
        plt.pause(SPEED) 
        
    plt.ioff()
    plt.close(fig)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":

    # Default fallbacks to prevent plotting errors if skipped
    history_full, food_full, len_full = [0], [0], [0]
    history_part, food_part, len_part = [0], [0], [0]

    # Create directory for saving weights
    weights_dir = "weightsA2C"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    choice = input("\nDo you want to train full, partial or both? [1, 2, 12]: ").strip().lower()
    
    if choice == '1':     
        print("Training Fully Observable Environment...")
        history_full, food_full, len_full = train_a2c(
            env_full, ac_full, optimizer_full, ITERATIONS_full, title="Fully Observable Training"
        )
        ac_full.save_weights(os.path.join(weights_dir, "ac_full.weights.h5"))
        print(f"✅ Saved Fully Observable weights to {weights_dir}/ac_full.weights.h5")

    elif choice == '2':
        print("Training Partially Observable Environment...")
        history_part, food_part, len_part = train_a2c(
            env_part, ac_part, optimizer_part, ITERATIONS_part, title="Partially Observable Training"
        )
        ac_part.save_weights(os.path.join(weights_dir, "ac_part.weights.h5"))
        print(f"✅ Saved Partially Observable weights to {weights_dir}/ac_part.weights.h5")
        
    elif choice == '12':
        print("Training Both Environments...")
        history_full, food_full, len_full = train_a2c(
            env_full, ac_full, optimizer_full, ITERATIONS_full, title="Fully Observable Training"
        )
        ac_full.save_weights(os.path.join(weights_dir, "ac_full.weights.h5"))
        print(f"✅ Saved Fully Observable weights to {weights_dir}/ac_full.weights.h5")
        
        history_part, food_part, len_part = train_a2c(
            env_part, ac_part, optimizer_part, ITERATIONS_part, title="Partially Observable Training"
        )
        ac_part.save_weights(os.path.join(weights_dir, "ac_part.weights.h5"))
        print(f"✅ Saved Partially Observable weights to {weights_dir}/ac_part.weights.h5")

    # ==============================================================================
    # FINAL VISUALIZATION
    # ==============================================================================
    
    # Define window sizes for smoothing the data to reduce noise in the plots
    window_r = 200
    window_m = 100
    
    # Smooth data for Fully Observable plots using a moving average (convolution).
    # Included a fallback to the original array if the array length is smaller than the window size.
    smoothed_r_full = np.convolve(history_full, np.ones(window_r)/window_r, mode='valid') if len(history_full) > window_r else history_full
    smoothed_f_full = np.convolve(food_full, np.ones(window_m), mode='valid') if len(food_full) > window_m else food_full
    smoothed_l_full = np.convolve(len_full, np.ones(window_m)/window_m, mode='valid') if len(len_full) > window_m else len_full
    
    # Smooth data for Partially Observable plots using a moving average.
    # Included a fallback to the original array if the array length is smaller than the window size.
    smoothed_r_part = np.convolve(history_part, np.ones(window_r)/window_r, mode='valid') if len(history_part) > window_r else history_part
    smoothed_f_part = np.convolve(food_part, np.ones(window_m), mode='valid') if len(food_part) > window_m else food_part
    smoothed_l_part = np.convolve(len_part, np.ones(window_m)/window_m, mode='valid') if len(len_part) > window_m else len_part

    # 1. Fully Observable: Average Reward
    fig1 = plt.figure(figsize=(6, 5))
    if hasattr(fig1.canvas, 'manager'):
        fig1.canvas.manager.set_window_title("Fully Obs: Average Reward")
    plt.plot(smoothed_r_full, color='blue')
    plt.title('Fully Obs: Average Reward')
    plt.grid(True)
    
    # 2. Fully Observable: Food / 100 Iters
    fig2 = plt.figure(figsize=(6, 5))
    if hasattr(fig2.canvas, 'manager'):
        fig2.canvas.manager.set_window_title("Fully Obs: Food / 100 Iters")
    plt.plot(smoothed_f_full, color='green')
    plt.title('Fully Obs: Food / 100 Iters')
    plt.grid(True)
    
    # 3. Fully Observable: Average Length
    fig3 = plt.figure(figsize=(6, 5))
    if hasattr(fig3.canvas, 'manager'):
        fig3.canvas.manager.set_window_title("Fully Obs: Average Length")
    plt.plot(smoothed_l_full, color='red')
    plt.title('Fully Obs: Average Length')
    plt.grid(True)
    
    # 4. Partially Observable: Average Reward
    fig4 = plt.figure(figsize=(6, 5))
    if hasattr(fig4.canvas, 'manager'):
        fig4.canvas.manager.set_window_title("Partially Obs: Average Reward")
    plt.plot(smoothed_r_part, color='blue')
    plt.title('Partially Obs: Average Reward')
    plt.grid(True)
    
    # 5. Partially Observable: Food / 100 Iters
    fig5 = plt.figure(figsize=(6, 5))
    if hasattr(fig5.canvas, 'manager'):
        fig5.canvas.manager.set_window_title("Partially Obs: Food / 100 Iters")
    plt.plot(smoothed_f_part, color='green')
    plt.title('Partially Obs: Food / 100 Iters')
    plt.grid(True)
    
    # 6. Partially Observable: Average Length
    fig6 = plt.figure(figsize=(6, 5))
    if hasattr(fig6.canvas, 'manager'):
        fig6.canvas.manager.set_window_title("Partially Obs: Average Length")
    plt.plot(smoothed_l_part, color='red')
    plt.title('Partially Obs: Average Length')
    plt.grid(True)
    
    # 7. Comparison: Average Reward
    fig7 = plt.figure(figsize=(6, 5))
    if hasattr(fig7.canvas, 'manager'):
        fig7.canvas.manager.set_window_title("Comparison: Average Reward")
    plt.plot(smoothed_r_full, label='Fully Obs')
    plt.plot(smoothed_r_part, label='Partially Obs')
    plt.title('Comparison: Average Reward')
    plt.legend()
    plt.grid(True)
    
    # 8. Comparison: Food / 100 Iters
    fig8 = plt.figure(figsize=(6, 5))
    if hasattr(fig8.canvas, 'manager'):
        fig8.canvas.manager.set_window_title("Comparison: Food / 100 Iters")
    plt.plot(smoothed_f_full, label='Fully Obs')
    plt.plot(smoothed_f_part, label='Partially Obs')
    plt.title('Comparison: Food / 100 Iters')
    plt.legend()
    plt.grid(True)
    
    # 9. Comparison: Average Length
    fig9 = plt.figure(figsize=(6, 5))
    if hasattr(fig9.canvas, 'manager'):
        fig9.canvas.manager.set_window_title("Comparison: Average Length")
    plt.plot(smoothed_l_full, label='Fully Obs')
    plt.plot(smoothed_l_part, label='Partially Obs')
    plt.title('Comparison: Average Length')
    plt.legend()
    plt.grid(True)


    plt.show(block=True)
