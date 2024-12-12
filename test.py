#%%
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from src.utils import two_hot

def loss(reward_logits, reward_pred, target) -> Tuple[torch.Tensor, float]:
    with torch.no_grad():
        mse_loss = F.mse_loss(reward_pred, target).item()

    target = two_hot(target, -5, 5, 16)
    reward_log_probs = F.log_softmax(reward_logits, dim=-1)
    categorical_loss = F.kl_div(reward_log_probs, target, reduction="batchmean", log_target=False)

    return categorical_loss, mse_loss

reward_logits = torch.randn(3, 16)
reward_pred = torch.randn(3)
target = torch.randn(3)

loss(reward_logits, reward_pred, target)
# %%
