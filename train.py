#%%
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import  DDIMScheduler

from src.dataloading import GymDataGenerator, gym_collate_function
from src.diffusion_model import DiffusionModel
from src.finetuning import modify_unet_for_multi_frame, lora_unet_for_multi_frame
from src.neural_env import NeuralEnv

#%%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    original_env = "LunarLander-v3"
    model_id="stabilityai/stable-diffusion-2-1"

    autoencoder = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device).requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
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
    dataset = GymDataGenerator(state_size, original_env, training_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=gym_collate_function, num_workers=8)


    diffusion = DiffusionModel(autoencoder, unet, diffusion_scheduler, state_size, dataset.env.action_space.n).to(device)
    neural_env = NeuralEnv(diffusion,dataset.env)
    
    # Optimizer
    learning_rate = 1e-3
    optimizer = optim.AdamW(diffusion.parameters(), lr=learning_rate)
    optimizer_scheduler = ExponentialLR(optimizer, gamma=0.997)
    
    losses, timesteps = [], []
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
        frames = frames.to(device)
        action = action.to(device)
        action_emb = diffusion.action_embedder(action)
        
        # process the inputs
        with torch.no_grad():
            latents = diffusion.frames_to_latents(frames)
            target_latent, context_latents = latents[:, -1], latents[:, :-1]
            context_latents, noise_aug_emb = diffusion.noise_context_latents(context_latents)
            input_latent, noise, timestep, target = diffusion.noise_target_latent(target_latent)

        # make a prediction
        prediction=diffusion(input_latent, context_latents, timestep, action_emb, noise_aug_emb)

        # Compute loss
        loss = nn.functional.mse_loss(prediction, target, reduction='none').mean(dim=(-1,-2,-3))
        losses += loss.cpu().tolist()
        timesteps += timestep.cpu().tolist()
        loss = loss.mean()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_scheduler.step()
        
        
        if step % 100 == 0 and step!=0:
            plt.close()
            plt.title('Training Loss Over Time')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.plot(range(len(losses)), losses)
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig('training_loss.png')

            plt.close()
            plt.title('loss-noise relashionship')
            plt.xlabel('timesteps')
            plt.ylabel('loss')
            history_size = 1000
            plt.scatter(timesteps[history_size:],losses[history_size:], s=0.5)
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig('loss_noise.png')

        
            plt.close()
            unet.save_pretrained(f'/tmp/checkpoints/{original_env}')
            torch.save(unet.state_dict(), f'/tmp/checkpoints/{original_env}_state_dict.pt')
            history_plot = neural_env.make_history_plot(grid_size=(6,6),num_inference_steps=16)
            plt.imshow(history_plot)
            os.makedirs("./video", exist_ok=True)
            plt.savefig(f"video/frame_history_{step}.png", dpi=300)
            plt.close()
        step += 1
        pbar.set_description(f"Loss: {loss:.3f}")
        
        

#%%
# Save the final plot
plt.savefig('training_loss.png')
unet.save_pretrained(f'/tmp/checkpoints/{original_env}')
print("Training complete. Final loss plot saved as 'training_loss.png'.")

# %%





# to load
# unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
# lora = lora_unet_for_multi_frame(unet, state_size, rank=128)
# lora.from_pretrained(unet,'/tmp/checkpoints/{original_env}')
