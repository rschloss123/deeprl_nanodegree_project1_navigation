
# Overview  

This project is part of the Value-Based Methods course in Udacity's Deep Reinforcement Learning Nanodegree. The goal of this project is to train a reinforcement learning agent in an environment similar to Unity's Banana Collector environment to collect yellow bananas and avoid blue bananas. 

## Environment
The agent's state is fully observable. The observation/state size is 37 and the action size is 4. The agent can select between the four discrete actions:

- `0` - move forward
- `1` - move backward
- `2` - move left
- `3` - move right

At each timestep, the agent receives a reward of +1 if it collects a yellow banana, and a reward of -1 if it collects a blue banana. The aim is to train the agent to select the best action so that it achieves an average reward of +13 over 100 consecutive episodes. 

## Agent
The agent is an implementation of the Deep-Q Network algorithm. Details will be discussed further in the included report.

# Installation

- To set up an Anaconda environment to run this code, you can follow the instructions included [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). These instructions include:

  - Creating a new environment
  - Installing OpenAI gym
  - Cloning Udacity's deep reinforcement learning repository and installing dependencies
  - Creating and running the IPython kernel in the Jupyter notebook that provides the solution to this project. 
- The Banana executable (Banana.exe) for the environment is included in this repository and works for Windows 10 (64 bit).

# Usage
- The solution to this project is provided in the Navigation notebook (Navigation.ipynb). Run each section of the notebook sequentially to train the agent. (You can skip Section 3, "Take Random Actions in the Environment," as this only displays random behavior of an untrained agent.)
- To view the details of the DQN implementation, see model.py and dqn_agent.py. 
