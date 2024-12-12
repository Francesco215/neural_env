#%%
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler,  StableDiffusionPipeline
from diffusers.schedulers import DDPMScheduler, DDIMScheduler

from src.hf_models import myUNet2DConditionModel
from src.dataloading import GymDataGenerator, gym_collate_function
from src.diffusion_model import DiffusionModel
from src.hf_schedulers import MyDPMSolverMultistepScheduler
from src.finetuning import modify_unet_for_multi_frame, lora_unet_for_multi_frame, save_conv_in_weights
from src.neural_env import NeuralEnv
from reward_classifier import RewardModel

#%%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # model_id="compvis/stable-diffusion-v1-4"
    model_id="stabilityai/stable-diffusion-2-1"

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
        unet = lora_unet_for_multi_frame(unet, state_size, rank=128)
    
    print(f"the scheduler prediction type is {diffusion_scheduler.config.prediction_type}")
    print(f"number of parameters in the autoencoder: {sum(p.numel() for p in autoencoder.parameters())//1e6}M")
    print(f"number of parameters in the unet: {sum(p.numel() for p in unet.parameters())//1e6}M")
    print(f"number of trainable parameters in the unet: {sum(p.numel() for p in unet.parameters() if p.requires_grad)//1e6}M")

    # Hyperparameters
    batch_size = 32
    training_steps = 10_000 * batch_size
    dataset = GymDataGenerator(state_size, "LunarLander-v3", training_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=gym_collate_function, num_workers=8)


    diffusion = DiffusionModel(autoencoder, unet, diffusion_scheduler, state_size, dataset.env.action_space.n).to(device)
    reward_model = RewardModel(unet, 4, -4, n_bins=16).to(device)
    neural_env = NeuralEnv(diffusion,dataset.env,reward_model)
    
    # Optimizer
    learning_rate = 1e-3
    optimizer = optim.AdamW(diffusion.parameters(), lr=learning_rate)
    optimizer_scheduler = ExponentialLR(optimizer, gamma=0.997)
    
    diffusion_losses = []
    reward_mse_losses = []
    reward_CE_losses = []
    step = 0
    #%%
    plt.figure(figsize=(10, 5))
    plt.title('Training Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.ion()  # Turn on interactive mode
    
    pbar = tqdm(dataloader,total=dataset.training_examples//batch_size)
    for batch in pbar:
        frames, action, reward = batch
        
        diffusion_mse, reward_CE, reward_mse = neural_env.losses(frames, action, reward)
        
        loss = diffusion_mse #+ reward_CE 

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_scheduler.step()
        
        diffusion_losses.append(diffusion_mse.item())
        reward_CE_losses.append(reward_CE.item())
        reward_mse_losses.append(reward_mse)
        
        if step % 100 == 0 and step > 0:
            plt.close()
            plt.title('Training Loss Over Time')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.plot(diffusion_losses, label='diffusion_mse')
            plt.plot(reward_CE_losses, label='reward_CE')
            plt.plot(reward_mse_losses, label='reward_mse')
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            plt.savefig('training_loss.png')
        
            plt.close()
            save_dir = f'/tmp/checkpoints/LunarLander-{step}'
            unet.save_pretrained(save_dir)
            save_conv_in_weights(unet, save_dir)

            history_plot = neural_env.make_history_plot(grid_size=(6,6),num_inference_steps=16)
            plt.imshow(history_plot)
            os.makedirs("./video", exist_ok=True)
            plt.savefig(f"video/frame_history_{step}.png", dpi=300)
            plt.close()

        step += 1
        pbar.set_description(f"Loss: {loss:.3f}, diffusion_mse: {diffusion_mse:.3f}, reward_mse: {reward_mse:.3f}")
        
        

#%%
# Save the final plot
plt.savefig('training_loss.png')
unet.save_pretrained('/tmp/checkpoints/LunarLander-v2')
print("Training complete. Final loss plot saved as 'training_loss.png'.")

# %%





# to load
# unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
# lora = lora_unet_for_multi_frame(unet, state_size, rank=16)
# lora.from_pretrained(unet,'/tmp/checkpoints/LunarLander-v2')