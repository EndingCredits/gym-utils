Collection of wrappers and utilities for the OpenAI Gym.

The intention behind this project is to standardise agent design to enable easier comparisons between agents, and to streamline agent development (and cut down on code re-use). 

This collection will include:
* Commonly used wrappers such as frame-skip, image processing, multi-observation states, random agent starts, etc.
* Wrappers for episode history and replay memory to avoid the agent having to handle these itself. agent can obtain samples for training, as well as calculating useful information such as n-step return, via these devices.
* An agent 'skeleton' which standardises the agent's inteface with its environment (amongst other things).
* Implementations of this skeleton for popular algorithms, such as A3C, in a depp-learning-library-agnostic fashion (users simply define their model set-up and training functions).
* Various other utilities (e.g. knn dictionary for algorithms such as Neural Episodic Control)
* Helper functions for automatically training/testing an agent.
