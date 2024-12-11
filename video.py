#%%
import os
import einops
import torch
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt

from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler, DDPMScheduler

from src.neural_env import NeuralEnv
from src.finetuning import modify_unet_for_multi_frame, lora_unet_for_multi_frame
from src.neural_env import NeuralEnv
from src.diffusion_model import DiffusionModel

#%%

with torch.no_grad():
    tmp_env = gym.make("LunarLander-v2", render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_size=4
    autoencoder = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
    lora = lora_unet_for_multi_frame(unet, state_size, rank=16)
    lora.from_pretrained(unet,'/tmp/checkpoints/LunarLander-v2')

    # scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    diffusion = DiffusionModel(autoencoder, unet, scheduler, state_size, tmp_env.action_space.n, device=device)
    ###
    # create a temporary variable with our env, which will use rgb_array as render mode. This mode is supported by the RecordVideo-Wrapper
    neural_env = NeuralEnv(diffusion,tmp_env)

    for i in tqdm(range(1)):
        action = neural_env.action_space.sample()  # agent policy that uses the observation and info
        neural_env.step(action, num_inference_steps=16)

    #%%
    # latents=neural_env.latent_history
    # mean,std = neural_env.latent_history[0,:3].mean(),neural_env.latent_history[0,:3].std()
    history_plot=diffusion.latents_to_frames(neural_env.latent_history) 

    plt.imshow(history_plot[0])
    os.makedirs("./video", exist_ok=True)
    plt.savefig(f"video/frame_history.png", dpi=300)
    plt.show()
