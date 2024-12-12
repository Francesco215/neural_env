#%%
import cv2
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

    # %%
    neural_env.reset()
    num_frames = 64

    for _ in tqdm(range(num_frames)):
        action = neural_env.action_space.sample()  
        neural_env.step(action, 16)
    frames = neural_env.model.latents_to_frames(neural_env.latent_history)[0]
    frames = einops.rearrange(frames, 'h (t w) c -> t h w c', w=256).astype('uint8')

    # Parameters
    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

    # Initialize VideoWriter
    video_writer = cv2.VideoWriter(output_path, fourcc, fps = 24, frame_size=tuple(frames.shape[1:3]))

    # Write frames to the video
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    # Release the writer
    video_writer.release()

    print(f"Video saved to {output_path}")
    # %%
