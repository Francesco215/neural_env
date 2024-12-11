import torch

import cv2
import gymnasium as gym

from torch.utils.data import IterableDataset

class GymDataGenerator(IterableDataset):
    def __init__(self, state_size=6, environment_name="CartPole-v1", training_examples=10_000):
        self.state_size = state_size

        self.env = gym.make(environment_name,render_mode="rgb_array")

        self.evolution_time = 10
        self.training_examples = training_examples

    def __iter__(self):
        terminated = True

        for _ in range(self.training_examples):
            if terminated:
                observation, _ = self.env.reset()
                terminated = False
                reward = 0
                action = 0
                frame_history = []
                # action_history = []
                step_count = 0
            else:
                action = self.env.action_space.sample()  # Random action
                _ , reward, terminated, _, _ = self.env.step(action)
            
            # action_history.append(action)
            # action_history = action_history[-self.state_size:]

            if step_count%self.evolution_time>=self.evolution_time-self.state_size-1:
                frame = self.env.render()
                frame = resize_image(frame)
                # frame_image = Image.fromarray(frame)
                frame_history.append(torch.tensor(frame))
                frame_history = frame_history[-self.state_size:]
            
            if step_count > 0 and step_count%self.evolution_time==0:  # Skip the first step as we don't have a previous state
                assert len(frame_history)>=self.state_size
                frames = torch.stack(frame_history[-self.state_size:])

                actions = torch.tensor(action).clone()
                # if step_count%self.evolution_time==self.evolution_time-1:

                yield frames, actions, torch.tensor(reward).clone()
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