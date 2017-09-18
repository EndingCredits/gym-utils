# Gym Utils
```
>>> Hi Elon, it's me - reinforcement learning enthusiast. I want to start building my own agents an...
<<< Use openai gym and roboschool. Everything is self explainatory.
>>> I already installed it. But Mnih et al. talk about frameskipping and action repeats.
<<< I don't have time for this: I have to go send a rocket to mars!
```
Bootstrapping your (deep) reinforcement learning agent with openai gym doesn't need to be awkward anymore.

## What is it?
Gym-utils is a collection of wrappers and utilities for use with the OpenAI Gym. The intention behind this project is to simplify and standardise agent design.

## Why is it important?
The OpenAI Gym standardises the reinforcement learning environment interface, making it easy to build agents that are applicable to a wide number of environments. This allows easy integration of new environments, and effortless switching between existing ones.

However, designing and testing new agents is often not so simple. One reason for this is that agents often feature minor small optimisations and input-processing steps such as frame-skip, and sequences of input observations. Not only is reimplementing these tedious, but individual implementations are often differ from each other (and errors are very easily introduced). Additionally, visually inspecting the differences between two algorithms can also often be challenging, since the relevant code is often obscured by these routines. Since one of the motivations behind the OpenAI Gym is to allow easy comparisons between different agents across environments, standardising these procedures is especially important.


## What's included?
* Commonly used wrappers such as frame-skip, image processing, multi-observation states, random agent starts, etc.
* Wrappers for episode history and replay memory. These wrappers provide automatic updating of episode hisoty, and replay memory, as well as providing useful functions such as calulating things like the n-step Q-return, these wrappers allow
* An agent 'skeleton' which standardises the agent's inteface with its environment (amongst other things).
* Implementations of this skeleton for popular algorithms, such as A3C, in a deep-learning-library-agnostic fashion (users simply define their model set-up and training functions).
* Various other utilities (e.g. knn dictionary for algorithms such as Neural Episodic Control).
* Helper functions for automatically training/testing an agent.

## Can I help?
Absolutely. As much as anything this is intended to be a collection of common resources, and if it's useful to you, then it's probably useful to others. If you modify, or reimplement, any part of this repository, please consider integrating your changes (although we would like to keep things as backwards compatible as possible) and submitting a pull request.


#### Notes and TODO list:
* A wrapper to automatically collect env statistics (e.g. episode rewards) would be useful. (It's possible this is already implemented in env monitoring)
* Need to look at pulling apart the Q agent from the tensorflow part.
* Need to define set supported observation types and add asserts where they don't work
