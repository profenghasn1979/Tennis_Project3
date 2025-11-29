# Collaboration and Competition (Tennis)

This project implements a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to solve the Unity ML-Agents Tennis environment.

## Project Details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

- **State Space**: The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Three stacked observations are used, resulting in a vector of 24 variables.
- **Action Space**: Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
- **Solution Criteria**: The environment is considered solved when the agents get an average score of +0.5 (over 100 consecutive episodes, taking the maximum score of the two agents per episode).

## Dependencies

- Python 3.6+
- PyTorch
- Unity ML-Agents
- NumPy
- Matplotlib

## Installation

1.  Clone this repository.
2.  Install the dependencies:
    ```bash
    pip install torch numpy matplotlib unityagents
    ```
3.  Download the Tennis environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/4028054/windows-10-repair-or-remove-programs-in-windows-10) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

4.  Place the file in the project folder and unzip (or decompress) the file.

## How to Run

1.  Navigate to the project directory.
2.  Run the training script:
    ```bash
    python main.py
    ```
3.  The script will print the average score every 100 episodes.
4.  When the environment is solved (avg score > 0.5), the model weights will be saved to `checkpoint_actor_0.pth`, `checkpoint_critic_0.pth`, etc.
