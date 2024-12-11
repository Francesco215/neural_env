from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
from torch import Tensor

class MyDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):
    def get_velocity(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """
        Computes the velocity (`v`) for `v_prediction` based on the original samples, noise, and timesteps.

        Args:
            original_samples (`torch.Tensor`):
                The original (clean) samples from the dataset.
            noise (`torch.Tensor`):
                The noise added to the original samples.
            timesteps (`torch.Tensor`):
                The current timesteps corresponding to each sample in the batch.

        Returns:
            `torch.Tensor`:
                The computed velocity tensor.
        """
        # Retrieve the corresponding sigma values for the given timesteps
        sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        sigma = sigmas[timesteps.clone().to('cpu')]

        # Compute alpha_t and sigma_t using the existing helper method
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        # Reshape alpha_t and sigma_t to enable broadcasting with original_samples and noise
        # Assumes original_samples and noise have shape (batch_size, channels, height, width)
        # Adjust the number of unsqueezes based on the number of dimensions
        alpha_t = alpha_t.view(-1, *([1] * (original_samples.ndim - 1))).to(noise.device)
        sigma_t = sigma_t.view(-1, *([1] * (original_samples.ndim - 1))).to(noise.device)

        # Compute velocity: v = alpha_t * x0 - sigma_t * noise
        velocity = alpha_t * original_samples - sigma_t * noise

        return velocity

