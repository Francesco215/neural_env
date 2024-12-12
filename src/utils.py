import torch
from torch import nn
from torch.nn import functional as F
import einops


@torch.jit.script
def symlog(x):
	return torch.sign(x) * torch.log(1 + torch.abs(x))

@torch.jit.script
def symexp(x):
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    
def bmult(x,y):
    if isinstance(y,float):
        return x*y
    return einops.einsum(x, y, 'b ..., b -> b ...')

def simnorm(z, V=8):
    shape = z.shape
    z = z.view(*shape[:-1], -1, V)
    z = F.softmax(z, dim=-1)
    return z.view(*shape)


def two_hot(x, vmin, vmax, num_bins):
    bin_size = (vmax - vmin) / (num_bins - 1) 
    x = torch.clamp(symlog(x), vmin, vmax)
    bin_idx = torch.floor((x - vmin) / bin_size).long()
    bin_offset = ((x - vmin) / bin_size - bin_idx.float())
    soft_two_hot = bmult(F.one_hot(bin_idx, num_bins).float(), 1 - bin_offset)
    soft_two_hot += bmult(F.one_hot((bin_idx + 1) % num_bins, num_bins).float(), bin_offset)
    return soft_two_hot

def two_hot_inv(x, vmin, vmax, num_bins):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    bins = torch.linspace(vmin, vmax, num_bins, device=x.device)
    x = torch.sum(x * bins, dim=-1)
    return x
    return symexp(x)

class SparseTransform(nn.Module):
    def __init__(self, V=8):
        super().__init__()

        self.V=V

    def forward(self, x):
        x = torch.mean(x,dim=(-1,-2))
        # x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)

        return simnorm(x, V=self.V)