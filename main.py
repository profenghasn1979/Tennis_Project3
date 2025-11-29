from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from maddpg_agent import Agent, ReplayBuffer, BUFFER_SIZE, BATCH_SIZE

def main():
    # 1. Initialize the Unity Environment
    # NOTE: Change file_name to match the location of the Tennis app on your machine
    env = UnityEnvironment(file_name="Tennis.app", worker_id=4)

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # Size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # Examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # 2. Initialize the Agents
    # Create two agents
    agent_0 = Agent(state_size, action_size, random_seed=0)
    agent_1 = Agent(state_size, action_size, random_seed=0)

    # Share Replay Buffer
    # We create a shared buffer and assign it to both agents
    shared_memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed=0)
    agent_0.memory = shared_memory
    agent_1.memory = shared_memory

    # 3. Training Loop
    n_episodes = 3000
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        agent_0.reset()                                        # reset the noise
        agent_1.reset()
        score = np.zeros(num_agents)                           # initialize the score (for each agent)
        
        while True:
            # Get actions from both agents
            action_0 = agent_0.act(states[0])
            action_1 = agent_1.act(states[1])
            actions = np.stack((action_0, action_1))           # combine actions
            
            # Step the environment
            # print("Action shape:", actions.shape)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            # Store experience and learn
            # Both agents add to the shared buffer and learn from it
            agent_0.step(states[0], action_0, rewards[0], next_states[0], dones[0])
            agent_1.step(states[1], action_1, rewards[1], next_states[1], dones[1])

            states = next_states
            score += rewards
            if np.any(dones):
                break
        
        # Record score (max of the two agents)
        max_score = np.max(score)
        scores_deque.append(max_score)
        scores.append(max_score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        # Save checkpoint if solved
        if np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            torch.save(agent_0.actor_local.state_dict(), 'checkpoint_actor_0.pth')
            torch.save(agent_0.critic_local.state_dict(), 'checkpoint_critic_0.pth')
            torch.save(agent_1.actor_local.state_dict(), 'checkpoint_actor_1.pth')
            torch.save(agent_1.critic_local.state_dict(), 'checkpoint_critic_1.pth')
            break
    
    env.close()

if __name__ == "__main__":
    main()
