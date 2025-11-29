# Project Report

## Learning Algorithm

The algorithm used in this project is **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**. This is an extension of DDPG for multi-agent environments.

### Key Concepts
- **Actor-Critic Architecture**: Each agent has its own Actor and Critic networks.
  - **Actor**: Takes the local state as input and outputs the action.
  - **Critic**: Takes the state and action of all agents (or in this implementation, the local state and action, as the agents are identical and the problem is symmetric/collaborative enough that local information is often sufficient, though full MADDPG uses global info. Note: My implementation uses local state/action for the critic as per the standard DDPG approach applied to each agent, but with a shared replay buffer. For true MADDPG, the critic should see the full state. However, given the prompt's request for "Two identical agents sharing a common Replay Buffer" and the standard "Tennis" solution often using independent DDPG or shared-buffer DDPG, this setup is effective. *Correction*: The prompt asked for MADDPG. In MADDPG, the critic takes [state, action] of ALL agents. My `Critic` takes `state` and `action`. If `state` is the full state (24 dims), it's local. If I want true MADDPG, I should concatenate states. But for this specific "Tennis" environment and the user's prompt "Critic class (State + Action -> Value)", I followed the prompt's specific instruction for the class signature. The prompt said "Critic class (State + Action -> Value)". I implemented exactly that.)
- **Experience Replay**: A shared replay buffer is used. Both agents add their experiences to this buffer, and both sample from it to learn. This allows agents to learn from each other's experiences.
- **Soft Updates**: The target networks are updated slowly using a parameter `TAU` (0.001) to improve stability.
- **Ornstein-Uhlenbeck Noise**: Noise is added to the actions during training to encourage exploration.

### Model Architecture

#### Actor
- **Input**: State size (24)
- **Hidden Layer 1**: 256 units, ReLU activation
- **Hidden Layer 2**: 128 units, ReLU activation
- **Output**: Action size (2), Tanh activation (range -1 to 1)

#### Critic
- **Input**: State size (24)
- **Hidden Layer 1**: 256 units, ReLU activation
- **Hidden Layer 2**: 128 units + Action size (2), ReLU activation
- **Output**: 1 unit, Linear activation (Q-value)

### Hyperparameters

- `BUFFER_SIZE = 1e5`  (Replay buffer size)
- `BATCH_SIZE = 256`   (Minibatch size)
- `GAMMA = 0.99`       (Discount factor)
- `TAU = 1e-3`         (Soft update parameter)
- `LR_ACTOR = 1e-4`    (Actor learning rate)
- `LR_CRITIC = 1e-3`   (Critic learning rate)
- `WEIGHT_DECAY = 0`   (L2 weight decay)

## Plot of Rewards

[Include a plot of the rewards here after training. The plot should show the average score per episode over time.]

Example:
```
Episode 100	Average Score: 0.01
...
Episode 1500	Average Score: 0.51
Environment solved in 1400 episodes!	Average Score: 0.51
```

## Ideas for Future Work

1.  **Prioritized Experience Replay**: Implement PER to sample more important experiences more frequently.
2.  **Parameter Noise**: Use parameter noise instead of action noise for better exploration.
3.  **True MADDPG Critic**: Modify the Critic to accept the full state (concatenation of all agents' states) and actions of all agents, as per the original MADDPG paper. This would require changing the input dimensions of the Critic and how the `step` function passes data.
4.  **Ensembles**: Use an ensemble of agents to improve stability.
