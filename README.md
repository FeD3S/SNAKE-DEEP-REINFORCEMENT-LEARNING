# Deep RL Snake: Navigating Fully and Partially Observable Environments

An advanced Reinforcement Learning project that trains AI agents to master the classic game of Snake. This repository explores the performance differences between various learning architectures and benchmark policies, specifically focusing on how agents adapt to fully observable versus partially observable environments.

Developed by Federico Saporiti, Control Systems Engineer, University of Paudua (Italy).

## Project Overview

The primary objective of this project is to implement and evaluate a **Double Deep Q-Network (DQN)** against a deterministic heuristic baseline and an **Advantage Actor-Critic (A2C)** architecture.

## Architectures Implemented

### 1. Heuristic Baseline
A naive greedy algorithm that calculates immediate coordinate displacement between the snake's head and the target fruit. It prioritizes horizontal alignment before vertical alignment. 
* **Limitation:** It lacks spatial awareness, ignores walls, and traps itself once the snake reaches 6 to 8 segments.

### 2. Double Deep Q-Network (DQN)
The core solution utilizes a Multi-Layer Perceptron (128, 256, and 64 units with ReLU activations). 
* **Exploration:** Employs an epsilon-greedy strategy. The minimum exploration rate ($\epsilon_{min}$) varies depending on the environment: 0.001 for fully observable (allowing high exploitation) and 0.1 for partially observable (forcing random exploration to break safe loops).
* **Action Selection vs. Evaluation:** Decouples action selection (online network) from evaluation (target network) to mitigate overestimation bias.

### 3. Advantage Actor-Critic (A2C)
A dual-headed network sharing an MLP base, outputting both policy logits (Actor) and state value estimates (Critic).
* **Exploration:** Discards epsilon-greedy methods for stochastic sampling directly from its learned policy distribution, regulated by an entropy bonus. 

## Training & Results

Models were trained over 80,000 iterations. 

* **Fully Observable Environments:** Both DQN and A2C demonstrated robust, monotonic convergence. The Double DQN achieved an average reward of roughly 0.50 and converged around a length of 17.5 to 18.5 segments.
* **Partially Observable Environments (5x5 ego-centric view):** Performance was characterized by high volatility and lower ceilings. Agents frequently adopted a conservative survival strategy—looping tightly upon themselves to avoid unseen collisions—which halted systematic growth and caused violent swings in average length (between 12.5 and 17.5 for DQN).

### Conclusion
The **Double DQN** emerged as the practically superior approach. While the A2C architecture achieved a comparable performance ceiling, the DQN avoids the developmental complexity of balancing policy gradient updates, value predictions, and entropy bonuses required by the dual-headed A2C model. 

Surviving the late game remains an extremely difficult spatial challenge for both architectures, as the snake's growing body severely clutters the board and punishes spatial miscalculations.
