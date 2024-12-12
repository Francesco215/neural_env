import torch
import random

import cv2
import gymnasium as gym

from torch.utils.data import IterableDataset

class GymDataGenerator(IterableDataset):
    def __init__(self, state_size=6, environment_name="CartPole-v1", training_examples=10_000):

        self.env = gym.make(environment_name,render_mode="rgb_array")

        self.state_size = state_size
        self.evolution_time = 50
        self.training_examples = training_examples

        assert self.evolution_time > self.state_size
    def __iter__(self):
        terminated = True
        training_examples = 0

        while training_examples<self.training_examples:
            if terminated:
                observation, _ = self.env.reset()
                terminated = False
                frame_history = []
                step_count = 0
                evolution_time = self.evolution_time

            action = self.env.action_space.sample()  # Random action
            _ , reward, terminated, _, _ = self.env.step(action)
            
            # action_history.append(action)
            # action_history = action_history[-self.state_size:]

            if step_count>=evolution_time-self.state_size-1:
                frame = self.env.render()
                frame = resize_image(frame)
                frame_history.append(torch.tensor(frame))
                frame_history = frame_history[-self.state_size:]
            
            if step_count==evolution_time:  # Skip the first step as we don't have a previous state
                assert len(frame_history)>=self.state_size
                frames = torch.stack(frame_history[-self.state_size:])

                actions = torch.tensor(action).clone()
                # if step_count%self.evolution_time==self.evolution_time-1:

                yield frames, actions, torch.tensor(reward).clone()
                training_examples+=1
                evolution_time+= self.evolution_time
                frame_history = []
                
            step_count += 1


def resize_image(image_array):
    # Check if the input array has the correct shape
    if image_array.shape != (400, 600, 3):
        raise ValueError("Input array must have shape (400, 600, 3)")
    
    # Resize the image using OpenCV
    resized_image = cv2.resize(image_array, (256, 256), interpolation=cv2.INTER_AREA)
    
    return resized_image 
def gym_collate_function(batch):
    frame_histories, action_histories, rewards = zip(*batch)
    padded_frames = torch.stack(frame_histories)
    padded_actions = torch.stack(action_histories)
    
    return padded_frames, padded_actions, torch.Tensor(rewards)