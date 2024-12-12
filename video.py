#%%
import os
import einops
import torch
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt

from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler, DDPMScheduler, DDIMScheduler

from reward_classifier import RewardModel
from src.hf_models import myUNet2DConditionModel
from src.neural_env import NeuralEnv
from src.finetuning import modify_unet_for_multi_frame, lora_unet_for_multi_frame, load_conv_in_weights
from src.neural_env import NeuralEnv
from src.diffusion_model import DiffusionModel
from src.dataloading import GymDataGenerator

#%%

with torch.no_grad():
    tmp_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id="stabilityai/stable-diffusion-2-1"

    pretrained_path='/tmp/checkpoints/LunarLander-100'
    autoencoder = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device).requires_grad_(False)
    unet = myUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    # diffusion_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, prediction_type='epsilon')
    diffusion_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    latent_channels = autoencoder.config.latent_channels
    state_size = 8  # Number of frames including the target frame
    in_channels = latent_channels * state_size
    
    full_finetune = False
    if full_finetune:
        unet = modify_unet_for_multi_frame(unet, state_size)
    else:
        lora = lora_unet_for_multi_frame(unet, state_size, rank=128)

    
    lora.from_pretrained(unet,pretrained_path)
    load_conv_in_weights(lora.model, pretrained_path)
    unet = lora
 
    print(f"the scheduler prediction type is {diffusion_scheduler.config.prediction_type}")
    print(f"number of parameters in the autoencoder: {sum(p.numel() for p in autoencoder.parameters())//1e6}M")
    print(f"number of parameters in the unet: {sum(p.numel() for p in unet.parameters())//1e6}M")
    print(f"number of trainable parameters in the unet: {sum(p.numel() for p in unet.parameters() if p.requires_grad)//1e6}M")
    

    training_steps = 10_000 * 30
    dataset = GymDataGenerator(state_size, "LunarLander-v3", training_steps)

    diffusion = DiffusionModel(autoencoder, unet, diffusion_scheduler, state_size, dataset.env.action_space.n).to(device)
    reward_model = RewardModel(unet, 4, -4, n_bins=16).to(device)
    neural_env = NeuralEnv(diffusion,dataset.env,reward_model)

    #%%
    
    history_plot = neural_env.make_history_plot(grid_size=(6,6),num_inference_steps=16)
    plt.imshow(history_plot)

    #%%
    # latents=neural_env.latent_history
    # mean,std = neural_env.latent_history[0,:3].mean(),neural_env.latent_history[0,:3].std()
    history_plot=diffusion.latents_to_frames(neural_env.latent_history) 

    plt.imshow(history_plot[0])
    os.makedirs("./video", exist_ok=True)
    plt.savefig(f"video/frame_history.png", dpi=300)
    plt.show()

# %%
