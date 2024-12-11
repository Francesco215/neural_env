# Gymnasium Env to Neural World Model

This repository reproduces [GameNGen](https://gamengen.github.io/****) and [DIAMOND](https://diamond-wm.github.io/)

The objective is to be able to train a neural world model of a [Gymnasium](https://gymnasium.farama.org/index.html) [Environment](https://gymnasium.farama.org/api/env/) in around 30 minutes with a single consumer graphics card.

## How it works 

The world model is a [LoRa](https://arxiv.org/abs/2106.09685) of [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1). This allows for much faster and efficient training.

The first convolutional layer of the diffusion model is expanded to be able to take in multiple frames

## Future plans

- Leverage the world model to train a policy
- Use the world and policy model to train a RL agent