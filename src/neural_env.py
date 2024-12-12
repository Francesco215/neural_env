from gymnasium import Env
import torch
from torch import nn, Tensor
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

from .reward_classifier import RewardModel
from .diffusion_model import DiffusionModel
from .dataloading import resize_image


class NeuralEnv(Env, nn.Module):
    def __init__(self, simulator:DiffusionModel, parent_env: Env, reward_model:RewardModel):
        nn.Module.__init__(self)
        self.simulator = simulator
        self.device = simulator.device
        self.state_size= simulator.state_size

        self.reward_model = reward_model

        self.parent_env = parent_env
        self.render_mode = "rgb_array"
        self.action_space = parent_env.action_space
        self.observation_space = parent_env.observation_space
        assert parent_env.render_mode == "rgb_array", "Environment must be set to render rgb_array"

        self.reset()

    def reset(self, *args, **kwargs):
        self.latent_history = self.generate_starting_frames()
        self.action_history = []

    @torch.no_grad()
    def render(self):
        last_frame = self.latent_history[0,-1]
        out_frame = self.simulator.latents_to_frames(last_frame[None,None])
        return out_frame[0]

    def step(self, action, num_inference_steps=4):
        observation, info = self.simulator.generate_latent(self.latent_history, action, num_inference_steps)
        reward, terminated, truncated = self.reward_function(info), self.termination_function(info), self.truncation_function(info)

        self.latent_history = torch.cat((self.latent_history,observation[None]),dim=1)

        self.action_history.append(action)

        return observation, reward, terminated, truncated, info

    #functions to implement
    def reward_function(self, info):
        return self.reward_model(info)

    def termination_function(self, info):
        return False

    def truncation_function(self, info):
        return False
        

    def generate_starting_frames(self)->Tensor:
        self.parent_env.reset()
        for _ in range(15):
            self.parent_env.step(self.parent_env.action_space.sample())
        frames = []
        for i in range(self.simulator.state_size-1):
            action = self.action_space.sample()  # Random action
            _ , reward, terminated, _, _ = self.parent_env.step(action)

            frame = self.parent_env.render()
            frame = resize_image(frame)
            frames.append(frame)
        frames = np.array(frames)
        frames = torch.tensor(frames,device=self.simulator.device,dtype=torch.float).unsqueeze(0)

        return self.simulator.frames_to_latents(frames)

    def make_history_plot(self,grid_size = (6,6),num_inference_steps=4):
        self.reset()
        num_frames_to_generate = grid_size[0]*grid_size[1]-self.state_size+1

        for _ in tqdm(range(num_frames_to_generate)):
            action = self.action_space.sample()  
            self.step(action, num_inference_steps)

        history_plot=self.simulator.latents_to_frames(self.latent_history) 
        history_plot=rearrange(history_plot,'b h (t w) c-> b (t h) w c', t=grid_size[1])
        self.parent_env.reset()
        return history_plot[0]

    def losses(self, frames, action, reward):
        frames = frames.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        action_emb = self.simulator.action_embedder(action)
        
        # process the inputs
        with torch.no_grad():
            latents = self.simulator.frames_to_latents(frames)
            target_latent, context_latents = latents[:, -1], latents[:, :-1]
            context_latents, noise_aug_emb = self.simulator.noise_context_latents(context_latents)
            input_latent, noise, timesteps, target = self.simulator.noise_target_latent(target_latent)

        # make a prediction
        latent_prediction, middle_activation = self.simulator.forward(input_latent, context_latents, timesteps, action_emb, noise_aug_emb)

        # Compute loss
        diffusion_mse = self.simulator.loss(latent_prediction, target)

        reward_pred_cat, reward_pred = self.reward_model.forward(middle_activation)
        reward_CE, reward_mse = self.reward_model.loss(reward_pred_cat, reward_pred, reward)

        return diffusion_mse, reward_CE, reward_mse
