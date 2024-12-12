
# GameNGen Reproduction
This repository contains an attempt to reproduce the "Diffusion Models Are Real-Time Game Engines" (GameNGen) paper by Valevski et al. using simpler reinforcement learning tasks like LunarLander and CartPole.
## Overview
The original GameNGen paper demonstrated that a neural model can simulate the complex game DOOM in real-time. Our project aims to apply similar techniques to simpler environments to better understand the approach and potentially extend it to other domains.
## Key Components
- RL Agent Training: We use PPO to train agents for LunarLander and CartPole environments.
- Data Collection: Gameplay trajectories from the trained agents are recorded for use in training the generative model.
- Generative Diffusion Model: A modified version of Stable Diffusion is trained to predict the next frame given past observations and actions.
- Real-time Inference: The trained model is used to simulate the environment in real-time.





## Getting Started

Install dependencies (list requirements)
Train RL agents for LunarLander and CartPole
Collect gameplay data from trained agents
Train the diffusion model on the collected data
Run real-time simulation using the trained model

Current Status
(Describe the current state of the project, what has been implemented, and what's still in progress)
Future Work

Experiment with different RL environments
Optimize for faster inference
Explore ways to extend the context length
Investigate methods for improving long-term consistency

Contributing
We welcome contributions! Please see our contributing guidelines for more information.
Citation
If you use this code in your research, please cite the original GameNGen paper:
(Include citation for the GameNGen paper)
License
(Specify the license for your project)