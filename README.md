# Gymnasium Env to Neural Env
This repository is inspired by [GameNGen](https://gamengen.github.io/****) and [DIAMOND](https://diamond-wm.github.io/) and is designed for researchers and practitioners interested in leveraging diffusion models to simulate dynamic environments.

The main objectives are

- Having as few lines of code as possible (~400 lines in the `src/` folder and ~100 for the `train.py` code)
- Making it run on consumer hardware and performing experiments in ~1 hour

It creates a Neural [Gymnasium](https://gymnasium.farama.org/index.html) [Environment](https://gymnasium.farama.org/api/env/) where the dynamics are determined by a diffusion-based world model.

### Simplified code
```python
# take a starting gym environment
original_env = gym.make("LunarLander-v3", render_mode="rgb_array")

# start with some pretrained model
model_id="stabilityai/stable-diffusion-2-1"
autoencoder = AutoencoderKL.from_pretrained(model_id, subfolder="vae").requires_grad_(False)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
diffusion_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

# define the diffusion model
diffusion = DiffusionModel(autoencoder, unet, diffusion_scheduler, state_size, original_env.action_space.n).to(device)

# and you can define the NeuralEnv like so
neural_env = NeuralEnv(diffusion,original_env)
```

The `NeuralEnv` class is a subclass of `gymnasium.Env` and inherits all its methods. The key difference is that `.step()`, `.render()`, and other functions are evaluated using the `diffusion` module.
This assures that the external API of the `NeuralEnv` class is the same at the one of any other `gymnasium.Env` class.

The `diffusion` model can be trained in ~1h to simulate the dynamics of the `gym.Env` it was trained on.

## Examples
Here is an example with 26M parameters LoRa trained in ~30min on a RTX3090. The first 8 frames are given as starting frames, and all of the others are generated 

![frame_history_2800](https://github.com/user-attachments/assets/161aa71e-e48c-4dcd-b97c-5009d7077b78)

There are still some artifacts and it's not perfect, but it demonstrates how leveraging pretrained models like Stable Diffusion allows for efficient adaptation to train world models that simulate dynamic environments.

Pretrained models already encode rich representations from extensive training, reducing the computational and data requirements for fine-tuning, and enabling faster convergence to high-quality results.

## What's under the hood

The world model is a [LoRa](https://arxiv.org/abs/2106.09685) of [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1). This allows for much faster and efficient training. The graph below shows the result of a ~30min run on a RTX3090

![training_loss](https://github.com/user-attachments/assets/98711319-24ee-4998-a00b-20d136e9589a)

The first convolutional layer of the diffusion model is expanded to be able to take in multiple frames by frame stacking (image below taken from [the DIAMOND paper](https://arxiv.org/pdf/2405.12399))
<img width="907" alt="Screenshot 2024-12-13 at 00 24 05" src="https://github.com/user-attachments/assets/8ebdd007-2954-4ac2-ae20-bdf522602fc3" />

## Installation

## Installation

To set up the repository, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Francesco215/neural_env.git
    cd neural_env
    ```
2. Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Optionally, set up a virtual environment:
    ```bash
    pip install uv swig  # you still need to have swig installed globally :(
    uv venv
    . .venv/bin/activate
    uv pip install -e .
    ```

## Running the Code
-  To train the model, run:
  ```bash
  python train.py
  ```

-  To generate videos, run:
  ```bash
  python video.py
  ```




## Future plans

- Add noise-depentend weighting to the loss function
- At the moment the world model is trained on random actions, it would be nice to train it on an expert agent
- Leverage the world model to train a reward model
- Use the world and reward model to train a bootstrapped RL agent
