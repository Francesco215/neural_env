#%%
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from peft.mixed_model import PeftMixedModel

from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler,  StableDiffusionPipeline
from diffusers.schedulers import DDPMScheduler, DDIMScheduler

from src.hf_models import myUNet2DConditionModel
from src.dataloading import GymDataGenerator, gym_collate_function
from src.diffusion_model import DiffusionModel
from src.hf_schedulers import MyDPMSolverMultistepScheduler
from src.finetuning import load_conv_in_weights, modify_unet_for_multi_frame, lora_unet_for_multi_frame, save_conv_in_weights
from src.neural_env import NeuralEnv
from src.reward_classifier import RewardModel
from src.mars_optimizer import MARS

#%%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # model_id="compvis/stable-diffusion-v1-4"
    model_id="stabilityai/stable-diffusion-2-1"

    autoencoder = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device).requires_grad_(False)
    unet = myUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device).requires_grad_(False)
    # diffusion_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, prediction_type='epsilon')
    diffusion_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    latent_channels = autoencoder.config.latent_channels
    state_size = 8  # Number of frames including the target frame
    in_channels = latent_channels * state_size
    
    full_finetune = False
    if full_finetune:
        unet = modify_unet_for_multi_frame(unet, state_size)
    else:
        unet = lora_unet_for_multi_frame(unet, state_size, rank=128)
    

    unet.load_state_dict(torch.load("/workspace/LunarLander-600/state_dict.pt"),strict=True)
    # unet.requires_grad_(False)

    print(f"the scheduler prediction type is {diffusion_scheduler.config.prediction_type}")
    print(f"number of parameters in the autoencoder: {sum(p.numel() for p in autoencoder.parameters())//1e6}M")
    print(f"number of parameters in the unet: {sum(p.numel() for p in unet.parameters())//1e6}M")
    print(f"number of trainable parameters in the unet: {sum(p.numel() for p in unet.parameters() if p.requires_grad)//1e6}M")

    # Hyperparameters
    batch_size = 128
    training_steps = 10_000 * batch_size
    dataset = GymDataGenerator(state_size, "LunarLander-v3", training_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=gym_collate_function, num_workers=8)


    diffusion = DiffusionModel(autoencoder, unet, diffusion_scheduler, state_size, dataset.env.action_space.n).to(device)
    # diffusion = torch.compile(diffusion)
    reward_model = RewardModel(unet, 6, -6, n_bins=16).to(device)
    neural_env = NeuralEnv(diffusion,dataset.env,reward_model)
    
    # Optimizer
    learning_rate = 1e-3
     
    parameters = neural_env.parameters()
    optimizer = MARS(parameters, lr=learning_rate)
    optimizer_scheduler = ExponentialLR(optimizer, gamma=0.975)
    
    
    diffusion_losses = []
    reward_mse_losses = []
    reward_KL_losses = []
    step = 0
    #%%
    plt.figure(figsize=(10, 5))
    plt.title('Training Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.ion()  # Turn on interactive mode
    
    micro_batch_size = 16  # Define the micro-batch size
    accumulation_steps = batch_size // micro_batch_size  # Calculate the number of accumulation steps
    max_grad_norm =.03

    pbar = tqdm(dataloader, total=dataset.training_examples // batch_size)
    for batch_number, batch in enumerate(pbar):
        optimizer.zero_grad()
        frames, action, reward = batch

        diffusion_loss, reward_CE_loss, reward_mse_loss = [], [], []
        for i in range(accumulation_steps):
            micro_frames = frames[i * micro_batch_size:(i + 1) * micro_batch_size]
            micro_action = action[i * micro_batch_size:(i + 1) * micro_batch_size]
            micro_reward = reward[i * micro_batch_size:(i + 1) * micro_batch_size]

            diffusion_mse, reward_KL, reward_mse = neural_env.losses(micro_frames, micro_action, micro_reward)

            loss = (reward_KL + diffusion_mse) / accumulation_steps  # Normalize the loss

            # Backpropagation
            loss.backward()

            diffusion_loss.append(diffusion_mse.item())
            reward_CE_loss.append(reward_KL.item())
            reward_mse_loss.append(reward_mse)

        if loss.item()<1.2:
            nn.utils.clip_grad_norm_(parameters,max_grad_norm)
        optimizer.step()
        optimizer_scheduler.step()

        diffusion_losses.append(np.mean(diffusion_loss))
        reward_KL_losses.append(np.mean(reward_CE_loss))
        reward_mse_losses.append(np.mean(reward_mse_loss))

        if step % 10 == 0 and step > 0:
            plt.close()
            plt.title('Training Loss Over Time')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            # plt.plot(diffusion_losses, label='diffusion_mse')
            plt.plot(reward_KL_losses, label='reward_CE')
            plt.plot(reward_mse_losses, label='reward_mse')
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            plt.savefig(f'training_loss.png')
            plt.close()
            # save_dir = f'/tmp/checkpoints/LunarLander-{step}'
            # with torch.no_grad():
            #     history_plot = neural_env.make_history_plot(grid_size=(6,6),num_inference_steps=16)
            # plt.imshow(history_plot)
            # os.makedirs("./video", exist_ok=True)
            # plt.savefig(f"video/frame_history_{step}.png", dpi=300)
            # plt.close()

        step += 1
        pbar.set_description(f"KL_loss: {reward_KL_losses[-1]:.3f}, diffusion_mse: {diffusion_losses[-1]:.3f}, reward_mse: {reward_mse_losses[-1]:.3f}")
# %%
