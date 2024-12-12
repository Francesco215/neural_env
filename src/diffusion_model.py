import torch
from torch import nn, Tensor
from torch.nn import functional as F
import einops

from .utils import bmult, SparseTransform
from .hf_models import myUNet2DConditionModel

from diffusers import AutoencoderKL, LMSDiscreteScheduler


class DiffusionModel(nn.Module):
    def __init__(self, autoencoder:AutoencoderKL, unet:myUNet2DConditionModel, scheduler:LMSDiscreteScheduler, state_size, action_space, device="cpu"):
        super().__init__()

        assert autoencoder.config.latent_channels*state_size == unet.config.in_channels, "Autoencoder latent channels must match UNet input channels times state size"
        self.autoencoder = autoencoder
        self.unet = unet
        self.scheduler = scheduler

        self.noise_augmentation_max = 0.7
        self.noise_embedder = nn.Embedding(32,unet.config.cross_attention_dim)

        self.action_embedder = nn.Embedding(action_space,unet.config.cross_attention_dim)

        self.state_size = state_size
        # self.transform_middle_activation = lambda x: x.mean(dim=(-1,-2))

    def forward(self, target_latent, context_latents, timesteps, action_emb, noise_aug_emb):
        """
        Args:
            target_latent (Tensor): Shape (batch_size, latent_channels, latent_height, latent_width)
            context_latents (Tensor): Shape (batch_size, time, latent_channels, latent_height, latent_width)
            timesteps (Tensor): Shape (batch_size,)
            action_emb (Tensor): Shape (batch_size, cross_attention_dim)
            noise_aug_emb (Tensor): Shape (batch_size, cross_attention_dim)

        Returns:
            noise_pred (Tensor): Shape (batch_size, latent_channels, latent_height, latent_width)
        """

        latents = torch.cat([context_latents, target_latent.unsqueeze(1)], dim=1).clone()
        latents = einops.rearrange(latents, 'b t c h w -> b (t c) h w')
        action_emb, noise_aug_emb = action_emb.unsqueeze(1), noise_aug_emb.unsqueeze(1)
        encoder_hidden_states = torch.cat([action_emb, noise_aug_emb], dim=1)
        noise_pred, middle_activation = self.unet(latents, timesteps, encoder_hidden_states)
        return noise_pred, middle_activation

    def loss(self,latent_prediction, target):
        return F.mse_loss(latent_prediction, target)

    @torch.no_grad()
    def noise_target_latent(self, target_latent):
        batch_size = target_latent.shape[0]

        noise = torch.randn_like(target_latent)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device, dtype=torch.long)

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise 
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(target_latent, noise, timesteps)
        else:
            raise NotImplementedError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        input_latent = self.scheduler.add_noise(target_latent, noise, timesteps)

        return input_latent, noise, timesteps, target

    @torch.no_grad()
    def noise_context_latents(self, context_latents):
        batch_size = context_latents.shape[0]

        #TODO: maybe find a better way of doing it
        noise_aug_level = torch.rand(batch_size, device=self.device) * self.noise_augmentation_max
        noise_aug_bucket = (noise_aug_level * self.noise_embedder.num_embeddings).long()
        noise_aug_emb = self.noise_embedder(noise_aug_bucket)
    
        context_noise = bmult(torch.randn_like(context_latents), noise_aug_level)
        context_latents = bmult(context_latents, 1 - noise_aug_level)
        context_latents = context_latents + context_noise

        return context_latents, noise_aug_emb

    @torch.no_grad()
    def generate_latent(self, context_latents, action, num_inference_steps=16, guidance=6.5) -> Tensor:
        context_latents = context_latents[:, -self.state_size+1:]

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        batch_size, time, channels, height, width = context_latents.shape
        latent_image = torch.randn((batch_size,channels,height,width), device=self.device) #make tensor of shape (batch_size, 1, channels, height, width)
        latent_image = latent_image * self.scheduler.init_noise_sigma
        
        # Prepare action embeddings
        action=torch.tensor(action).to(self.device)
        action_emb = self.action_embedder(action)[None]
        noise_aug_emb = self.noise_embedder(torch.tensor(1).to(self.device)).view(action_emb.shape)

        uncond_emb_1 = torch.zeros((batch_size, self.unet.config.cross_attention_dim), device=self.device)
        uncond_emb_2 = torch.zeros((batch_size, self.unet.config.cross_attention_dim), device=self.device)

        # Sampling loop
        for t in self.scheduler.timesteps:
            model_input = self.scheduler.scale_model_input(latent_image.clone(), timestep=t)
            noise_pred, middle_activation = self.forward(model_input, context_latents, t, action_emb, noise_aug_emb) 
            uncond_pred, uncond_middle_activation = self.forward(model_input, context_latents, t, uncond_emb_1, uncond_emb_2)
            # Scheduler step

            noise_pred = uncond_pred + guidance * (noise_pred - uncond_pred)
            latent_image = self.scheduler.step(noise_pred, t, latent_image).prev_sample

        return latent_image, middle_activation

    @property 
    def device(self):
        return next(self.parameters()).device
    
    @torch.no_grad()
    def frames_to_latents(self, frames)->Tensor:
        """
        frames.shape: (batch_size, time, height, width, rgb)
        latents.shape: (batch_size, time, latent_channels, latent_height, latent_width)
        """
        batch_size = frames.shape[0]

        frames = frames / 127.5 - 1  # Normalize from (0,255) to (-1,1)
        frames = einops.rearrange(frames, 'b t h w c -> (b t) c h w')

        #split the conversion to not overload the GPU RAM
        split_size = 64
        for i in range (0, frames.shape[0], split_size):
            l = self.autoencoder.encode(frames[i:i+split_size]).latent_dist.sample()
            if i == 0:
                latents = l
            else:
                latents = torch.cat((latents, l), dim=0)

        # Apply scaling factor
        latents = latents * self.autoencoder.config.scaling_factor
        latents = einops.rearrange(latents, '(b t) c h w -> b t c h w', b=batch_size)
        return latents
        
    def latents_to_frames(self,latents):
        """
            Converts latent representations to frames.
            Args:
                latents (torch.Tensor): A tensor of shape (batch_size, time, latent_channels, latent_height, latent_width) 
                                        representing the latent representations.
            Returns:
                numpy.ndarray: A numpy array of shape (batch_size, height, width * time, rgb) representing the decoded frames.
            Note:
                - The method uses an autoencoder to decode the latent representations.
                - The frames are rearranged and clipped to the range [0, 255] before being converted to a numpy array.
        """
        batch_size = latents.shape[0]
        latents = einops.rearrange(latents, 'b t c h w -> (b t) c h w')
        # Apply inverse scaling factor
        latents = latents / self.autoencoder.config.scaling_factor
        frames = self.autoencoder.decode(latents).sample
        frames = einops.rearrange(frames, '(b t) c h w -> b h (t w) c', b=batch_size)
        frames = torch.clip((frames + 1) * 127.5, 0, 255).cpu().detach().numpy().astype(int)
        return frames

        