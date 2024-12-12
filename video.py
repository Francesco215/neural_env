#%%
import os
import einops
import torch
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

from src.neural_env import NeuralEnv
from src.finetuning import modify_unet_for_multi_frame, lora_unet_for_multi_frame
from src.neural_env import NeuralEnv
from src.diffusion_model import DiffusionModel

#%%

with torch.no_grad():
    original_env = "LunarLander-v3"
    model_id="stabilityai/stable-diffusion-2-1"

    tmp_env = gym.make(original_env, render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_size=8
    autoencoder = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    diffusion_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    lora = lora_unet_for_multi_frame(unet, state_size, rank=128)
    lora.load_state_dict(torch.load("/tmp/checkpoints/LunarLander-v3_state_dict.pt"),strict=True)

    diffusion = DiffusionModel(autoencoder, unet, diffusion_scheduler, state_size, tmp_env.action_space.n).to(device)
    # create a temporary variable with our env, which will use rgb_array as render mode. This mode is supported by the RecordVideo-Wrapper
    neural_env = NeuralEnv(diffusion,tmp_env)

    #%% 
    history_plot = neural_env.make_history_plot(grid_size=(6,6),num_inference_steps=16)

    #%%
    plt.imshow(history_plot)
    os.makedirs("./video", exist_ok=True)
    plt.savefig(f"video/frame_history.png", dpi=300)
    plt.show()

# %%
