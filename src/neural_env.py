from gymnasium import Env
import torch
from torch import nn, Tensor
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

from .diffusion_model import DiffusionModel
from .dataloading import resize_image


class NeuralEnv(Env):
    def __init__(self, model:DiffusionModel, parent_env: Env):
        self.model = model
        self.state_size=model.state_size
        
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
        out_frame = self.model.latents_to_frames(last_frame[None,None])
        return out_frame[0]

    def step(self, action, num_inference_steps=4):
        observation = self.model.generate_latent(self.latent_history, action, num_inference_steps)
        reward = self.reward_function(observation)
        terminated = self.termination_function(observation)
        truncated, info = False, None

        self.latent_history = torch.cat((self.latent_history,observation[None]),dim=1)

        self.action_history.append(action)

        return observation, reward, terminated, truncated, info

    #functions to implement
    def reward_function(self, observation):
        return 0

    def termination_function(self, observation):
        return False
        

    def generate_starting_frames(self)->Tensor:
        self.parent_env.reset()
        for _ in range(15):
            self.parent_env.step(self.parent_env.action_space.sample())
        frames = []
        for i in range(self.model.state_size-1):
            action = self.action_space.sample()  # Random action
            _ , reward, terminated, _, _ = self.parent_env.step(action)

            frame = self.parent_env.render()
            frame = resize_image(frame)
            frames.append(frame)
        frames = np.array(frames)
        frames = torch.tensor(frames,device=self.model.device,dtype=torch.float).unsqueeze(0)

        return self.model.frames_to_latents(frames)
            

    def make_history_plot(self,grid_size = (6,6),num_inference_steps=4):
        self.reset()
        num_frames_to_generate = grid_size[0]*grid_size[1]-self.state_size+1

        for _ in tqdm(range(num_frames_to_generate)):
            action = self.action_space.sample()  
            self.step(action, num_inference_steps)

        history_plot=self.model.latents_to_frames(self.latent_history) 
        history_plot=rearrange(history_plot,'b h (t w) c-> b (t h) w c', t=grid_size[1])
        self.parent_env.reset()
        return history_plot[0]
