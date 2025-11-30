# Project Report

## Learning Algorithm

The algorithm used in this project is **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**. This is an extension of DDPG for multi-agent environments.

### Key Concepts
- **Actor-Critic Architecture**: Each agent has its own Actor and Critic networks.
  - **Actor (Policy)**: Takes the local state as input and outputs the action. It is a deterministic policy $\mu(s)$.
  - **Critic (Value)**: Takes the state and action of all agents (or in this implementation, the local state and action) and outputs the Q-value. It estimates the expected return $Q(s, a)$.
- **Centralized Training, Decentralized Execution**: This is the core of MADDPG. During training, the Critic can see extra information (like the actions of other agents), but during execution (testing), the Actor only needs its own local observation.
- **Experience Replay**: A shared replay buffer is used. Both agents add their experiences to this buffer, and both sample from it to learn. This allows agents to learn from each other's experiences and breaks the correlation between consecutive samples.
- **Soft Updates**: The target networks are updated slowly using a parameter `TAU` (0.001) to improve stability. $\theta_{target} = \tau \theta_{local} + (1 - \tau) \theta_{target}$.
- **Ornstein-Uhlenbeck Noise**: Correlated noise is added to the actions during training to encourage exploration in physical environments (like inertia).

### Hyperparameter Tuning & Improvements
To solve the environment, several key improvements were made to the standard DDPG baseline:

1.  **Batch Normalization**: Added `BatchNorm1d` layers to the Actor and Critic.
    *   *Why*: Deep RL is sensitive to the scale of inputs. Batch Norm standardizes the inputs to each layer, preventing gradients from vanishing or exploding and allowing for higher learning rates.
2.  **Gradient Clipping**: Clipped critic gradients to 1.0.
    *   *Why*: In the early stages, the Critic's error can be huge, leading to massive updates that destabilize the Actor. Clipping keeps updates reasonable.
3.  **Noise Decay**: Implemented a decay rate of 0.999 for the exploration noise.
    *   *Why*: Initially, agents need to explore wildly. As they learn, noise becomes a distraction. Decaying it allows them to fine-tune their policy (exploit).
4.  **Update Frequency**: Updated the networks every 2 steps, with 2 updates per step.
    *   *Why*: Updating every single step can be unstable. Accumulating a bit more experience before updating helps average out the noise in the gradients.

### Model Architecture

#### Actor
- **Input**: State size (24)
- **Hidden Layer 1**: 256 units + Batch Norm + ReLU
- **Hidden Layer 2**: 128 units + ReLU
- **Output**: Action size (2) + Tanh (range -1 to 1)

#### Critic
- **Input**: State size (24)
- **Hidden Layer 1**: 256 units + Batch Norm + ReLU
- **Hidden Layer 2**: 128 units + Action size (2) + ReLU
- **Output**: 1 unit (Linear Q-value)

### Hyperparameters

- `BUFFER_SIZE = 1e5`
- `BATCH_SIZE = 512` (Increased from 256 for stability)
- `GAMMA = 0.99`
- `TAU = 1e-3`
- `LR_ACTOR = 1e-4`
- `LR_CRITIC = 1e-3`
- `UPDATE_EVERY = 2`
- `NUM_UPDATES = 2`
- `NOISE_DECAY = 0.999`



## Plot of Rewards

The environment was solved in **1348 episodes**!

```
Episode 100     Average Score: 0.00
Episode 200     Average Score: 0.01
Episode 300     Average Score: 0.03
Episode 400     Average Score: 0.00
Episode 500     Average Score: 0.00
Episode 600     Average Score: 0.00
Episode 700     Average Score: 0.01
Episode 800     Average Score: 0.08
Episode 900     Average Score: 0.09
Episode 1000    Average Score: 0.10
Episode 1100    Average Score: 0.11
Episode 1200    Average Score: 0.11
Episode 1300    Average Score: 0.35
Episode 1400    Average Score: 0.24
Episode 1448    Average Score: 0.51
Environment solved in 1348 episodes!    Average Score: 0.51
```

## Ideas for Future Work

1.  **Prioritized Experience Replay**: Implement PER to sample more important experiences more frequently.
2.  **Parameter Noise**: Use parameter noise instead of action noise for better exploration.
3.  **True MADDPG Critic**: Modify the Critic to accept the full state (concatenation of all agents' states) and actions of all agents, as per the original MADDPG paper. This would require changing the input dimensions of the Critic and how the `step` function passes data.
4.  **Ensembles**: Use an ensemble of agents to improve stability.
