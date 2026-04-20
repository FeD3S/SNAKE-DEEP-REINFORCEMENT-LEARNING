import environments_fully_observable 
import environments_partially_observable
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

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
        states[i] = padded_boards[i, hx - self.mask_size : hx + self.mask_size + 1, hy - self.mask_size : hy + self.mask_size + 1]
                                  
    return tf.keras.utils.to_categorical(states, num_classes=5)[..., 1:]

environments_partially_observable.OriginalSnakeEnvironment.to_state = partial_to_state

# ==============================================================================
# NEURAL NETWORK ARCHITECTURES
# ==============================================================================
def build_global_mlp_network(input_shape):
    """
    Rebuilds the Multi-Layer Perceptron (MLP) for Epsilon-Greedy (DQN) weight loading.
    Outputs a single tensor of Q-values.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    q_values = tf.keras.layers.Dense(4)(x)
    return tf.keras.Model(inputs=inputs, outputs=q_values)

def build_actor_critic_network(input_shape):
    """
    Rebuilds the Actor-Critic network for A2C weight loading.
    Outputs a list of two tensors: [logits, value].
    """
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
# UTILITY FUNCTIONS
# ==============================================================================
def get_action_mask(env):
    """
    Generates a binary mask to prevent the agent from intentionally moving into walls 
    or its own body during the evaluation.
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
# INFERENCE / EVALUATION LOOP
# ==============================================================================
def visualize_game(env, model, policy_type, steps=300, title="Trained Agent Evaluation", speed=0.05):
    """
    Runs a fully trained agent purely on exploitation.
    Handles both Q-networks (outputs single tensor) and Actor-Critic networks (outputs list).
    """
    plt.ion() 
    fig, ax = plt.subplots(figsize=(6, 6))
    
    if hasattr(fig.canvas, 'manager'):
        fig.canvas.manager.set_window_title(title)
    
    for step in range(steps):
        # 1. Fetch current state 
        state = tf.constant(env.to_state(), dtype=tf.float32)
        
        # 2. Predict outputs based on the policy type
        output = model(state)
        
        # Isolate the logits/q-values depending on architecture
        # A2C returns [logits, values], so we take index 0
        # DQN returns just q_values, so we take the raw output
        if policy_type == "a2c":
            action_values = output[0] 
        else: # "dqn"
            action_values = output
        
        # 3. Apply the valid move mask to strictly forbid invalid moves
        mask = get_action_mask(env)
        masked_values = action_values + (mask - 1.0) * 1e9
        
        # 4. Pure exploitation: Select the action with the highest Q-value / Logit
        action = tf.argmax(masked_values, axis=-1, output_type=tf.int32)
        
        # 5. Advance the environment state
        env.move(action)
        
        # 6. Render the result
        ax.clear()
        ax.imshow(env.boards[0], origin="lower", cmap="viridis") 
        ax.set_title(f"{title} - Step {step+1} / {steps}")
        ax.axis('off')
        
        plt.draw()
        plt.pause(speed) 
        
    plt.ioff()
    plt.close(fig)
    print(f"Evaluation finished for: {title}")

# ==============================================================================
# EVALUATION HELPER FUNCTION
# ==============================================================================
def evaluate_policy(policy_type, base_dir, n_boards=1, board_size=10, steps=300):
    """
    Helper function to sequentially run Both Full and Partial environments
    for a specific policy type (A2C or Epsilon-Greedy).
    """
    # 1. Setup the directory and specific weight filenames based on the policy type
    if policy_type == "a2c":
        weights_dir = os.path.join(base_dir, 'weightsA2C')
        full_weights_name = 'ac_full.weights.h5'
        part_weights_name = 'ac_part.weights.h5'
        policy_label = "Actor-Critic"
    else: # "dqn"
        weights_dir = os.path.join(base_dir, 'weights')
        full_weights_name = 'q_full.weights.h5'
        part_weights_name = 'q_part.weights.h5'
        policy_label = "Epsilon-Greedy (DQN)"

    # Verify directory exists
    if not os.path.exists(weights_dir):
        print(f"Error: Directory '{weights_dir}' not found for {policy_label}.")
        return

    # --------------------------------------------------------------------------
    # EVALUATE FULLY OBSERVABLE
    # --------------------------------------------------------------------------
    print(f"\n--- Loading {policy_label} [Fully Observable] ---")
    env_full = environments_fully_observable.OriginalSnakeEnvironment(n_boards, board_size)
    
    # Build the correct network type for a 10x10 map
    if policy_type == "a2c":
        model_full = build_actor_critic_network((board_size, board_size, 4))
    else:
        model_full = build_global_mlp_network((board_size, board_size, 4))
        
    weight_path_full = os.path.join(weights_dir, full_weights_name)
    
    if os.path.exists(weight_path_full):
        model_full.load_weights(weight_path_full)
        visualize_game(env_full, model_full, policy_type=policy_type, steps=steps, title=f"{policy_label}: Fully Observable")
    else:
        print(f"Weights file not found at {weight_path_full}.")

    # --------------------------------------------------------------------------
    # EVALUATE PARTIALLY OBSERVABLE
    # --------------------------------------------------------------------------
    print(f"\n--- Loading {policy_label} [Partially Observable] ---")
    mask_size = 2
    env_part = environments_partially_observable.OriginalSnakeEnvironment(n_boards, board_size, mask_size=mask_size)
    
    # Build the correct network type for a 5x5 map (based on mask_size 2)
    grid_dim = 2 * mask_size + 1
    if policy_type == "a2c":
        model_part = build_actor_critic_network((grid_dim, grid_dim, 4))
    else:
        model_part = build_global_mlp_network((grid_dim, grid_dim, 4))
        
    weight_path_part = os.path.join(weights_dir, part_weights_name)
    
    if os.path.exists(weight_path_part):
        model_part.load_weights(weight_path_part)
        visualize_game(env_part, model_part, policy_type=policy_type, steps=steps, title=f"{policy_label}: Partially Observable")
    else:
        print(f"Weights file not found at {weight_path_part}.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Dynamically determine the absolute path to ensure weights are found reliably
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Prompt user for which policy to evaluate (removes full/partial choice as requested)
    choice = input("\nWhich policy do you want to evaluate? Actor-Critic (1), Epsilon-Greedy (2) or Both (12)?: ").strip()
    
    N_BOARDS = 1
    BOARD_SIZE = 10
    STEPS = 300

    # Evaluate based on choice. Full and Partial are run automatically inside evaluate_policy.
    if choice == '1':
        evaluate_policy(policy_type="a2c", base_dir=BASE_DIR, n_boards=N_BOARDS, board_size=BOARD_SIZE, steps=STEPS)

    elif choice == '2':
        evaluate_policy(policy_type="dqn", base_dir=BASE_DIR, n_boards=N_BOARDS, board_size=BOARD_SIZE, steps=STEPS)
            
    elif choice == '12':
        # Run A2C first, then Epsilon-Greedy
        evaluate_policy(policy_type="a2c", base_dir=BASE_DIR, n_boards=N_BOARDS, board_size=BOARD_SIZE, steps=STEPS)
        evaluate_policy(policy_type="dqn", base_dir=BASE_DIR, n_boards=N_BOARDS, board_size=BOARD_SIZE, steps=STEPS)
    else:
        print("Invalid choice. Please select 1, 2, or 12.")