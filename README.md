# Continuous Control

### Introduction 

This code solves Unity's [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. The goal is to move a double-jointed arm to a target location. 

Solution to the second project of Udacity's [Deep Reinforcement Learning](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) Nanodegree.

### Setting

A reward +0.1 is given for each time step where the agent's "hand" is in the goal position. It is thus the goal of the agent to track the moving target region as closely as possible, for as long as possible.

The observation space size is 33. The continuous action space size is 4, where each action corresponds to a value in [-1,1].  

In the current version, a single actor is to be trained. The environment is solved with an average reward of +30 over 100 consecutive periods.  

### Install dependencies

For correct installation, please make sure to use Python 3.6. 

In order to run the Jupyter Notebook `Continuous_Control.ipynb`, please see the installation instructions [here](https://jupyter.readthedocs.io/en/latest/install.html).   

To run the notebook, you also have to download the `Reacher` environment from Udacity's [project page](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control). If your system is not Linux, please adjust the respective line of code in the beginning of `Continuous_Control.ipynb` to point to your environment.


