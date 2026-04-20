import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import environments_fully_observable

def simple_greedy_policy(env):
    """
    A naive greedy baseline. 
    It calculates the displacement between the head and the fruit and picks the 
    first move that reduces that distance.
    """
    actions = []
    
    # Iterate over each parallel board
    for i in range(env.n_boards):
        board = env.boards[i]
        
        # 1. Locate Head
        head_coords = np.argwhere(board == env.HEAD)
        if len(head_coords) == 0:
            actions.append(env.UP)
            continue
        hx, hy = head_coords[0]
        
        # 2. Locate Fruit
        fruit_coords = np.argwhere(board == env.FRUIT)
        if len(fruit_coords) == 0:
            actions.append(env.UP)
            continue
        fx, fy = fruit_coords[0]
        
        # 3. Simple Coordinate Logic
        # Priority: Move on X axis first, then Y axis.
        # Environment axes mapping: UP(0): dx=1, RIGHT(1): dy=1, DOWN(2): dx=-1, LEFT(3): dy=-1
        if fx > hx:
            move = env.UP
        elif fx < hx:
            move = env.DOWN
        elif fy > hy:
            move = env.RIGHT
        elif fy < hy:
            move = env.LEFT
        else:
            move = env.UP # Already on fruit or trapped
            
        actions.append(move)
            
    return np.array(actions)

# ==============================================================================
# EVALUATION WITH LIVE PLOTTING
# ==============================================================================
def evaluate_baseline(n_boards=100, board_size=10, iterations=1000):
    env = environments_fully_observable.OriginalSnakeEnvironment(n_boards, board_size)
    
    rewards_history = []
    food_history = []
    length_history = []
    
    plt.ion() 
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    fig.canvas.manager.set_window_title("Naive Greedy Baseline (No Self-Avoidance)")
    
    for iteration in trange(iterations):
        # Get greedy actions
        actions = simple_greedy_policy(env)
        
        # Step environment
        rewards = env.move(actions)
        
        # Metrics
        food_history.append(np.sum(rewards.numpy() > 0) / env.n_boards)
        length_history.append(np.mean([len(b) + 1 for b in env.bodies]))
        rewards_history.append(np.mean(rewards.numpy()))

        # Live Update
        if iteration % 2 == 0:
            ax1.clear()
            ax1.imshow(env.boards[0], origin="lower")
            ax1.set_title(f"Board 0 - Iter {iteration}")
            ax1.axis('off')

            ax2.clear()
            ax2.plot(rewards_history[-200:], color='blue')
            ax2.set_title("Recent Rewards")
            
            ax3.clear()
            ax3.plot(np.cumsum(food_history), color='green')
            ax3.set_title("Total Food Eaten")
            
            ax4.clear()
            ax4.plot(length_history, color='red')
            ax4.set_title("Avg Snake Length")
            
            plt.draw()
            plt.pause(0.1) 

    plt.ioff()
    plt.show(block=True)

if __name__ == "__main__":
    evaluate_baseline()