from unityagents import UnityEnvironment
import numpy as np
import torch
from maddpg_agent import Agent

def watch():
    # 1. Initialize the Unity Environment
    # NOTE: Change file_name to match the location of the Tennis app on your machine
    # Ensure worker_id is different from training to avoid conflicts if training is still running
    env = UnityEnvironment(file_name="Tennis.app", worker_id=10)

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # Number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # Size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # Examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    # 2. Initialize the Agents
    agent_0 = Agent(state_size, action_size, random_seed=0)
    agent_1 = Agent(state_size, action_size, random_seed=0)

    # 3. Load the weights
    print("Loading weights...")
    agent_0.actor_local.load_state_dict(torch.load('checkpoint_actor_0.pth'))
    agent_1.actor_local.load_state_dict(torch.load('checkpoint_actor_1.pth'))
    print("Weights loaded successfully.")

    # 4. Watch them play
    for i in range(5): # Play 5 episodes
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        score = np.zeros(num_agents)                           # initialize the score (for each agent)
        
        while True:
            # Get actions from both agents (no noise for inference)
            action_0 = agent_0.act(states[0], add_noise=False)
            action_1 = agent_1.act(states[1], add_noise=False)
            actions = np.stack((action_0, action_1))           # combine actions
            
            # Step the environment
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            score += rewards
            states = next_states
            if np.any(dones):
                break
        
        print('Episode {}: Score: {}'.format(i+1, np.max(score)))

    env.close()

if __name__ == "__main__":
    watch()
