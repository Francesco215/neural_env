import torch
from torch import nn, Tensor
from torch.nn import functional as F


import einops
from typing import Tuple

from .hf_models import myUNet2DConditionModel
from .utils import two_hot_inv, two_hot, symlog

class RewardModel(nn.Module):
    def __init__(self, unet: myUNet2DConditionModel, max_reward:float, min_reward:float, n_bins:int=32):
        super().__init__()
        self.unet = unet

        self.middle_block_size = self.unet.mid_block.in_channels #the out_channels are the same as the in_channels

        self.mlp = MLP(in_dim=self.middle_block_size, out_dim=n_bins, hidden_dim=256)
        # self.mlp = nn.Linear(self.middle_block_size, n_bins)
        self.loss_function = nn.KLDivLoss(reduction="batchmean", log_target=False)

        self.max_reward, self.min_reward, self.n_bins = max_reward, min_reward, n_bins 


    def forward(self, sparse_encoding:Tensor):
        reward_logits = self.mlp(sparse_encoding)
        
        with torch.no_grad():
            reward_prob = F.softmax(reward_logits, dim=-1)
            float_reward = two_hot_inv(reward_prob, self.min_reward, self.max_reward, self.n_bins) 

        return reward_logits, float_reward

    def loss(self, reward_logits, reward_pred, target) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            mse_loss = F.mse_loss(reward_pred, symlog(target)).item()

        target = two_hot(target, self.min_reward, self.max_reward, self.n_bins)
        reward_log_probs = F.log_softmax(reward_logits, dim=-1)
        categorical_loss = self.loss_function(reward_log_probs, target)

        return categorical_loss, mse_loss


# class PolicyModel(nn.Module):
#     def __init__(self, neural_env:NeuralEnv):
#         super().__init__()
#         self.neural_env = neural_env
#         self.unet = neural_env.simulator
#         self.action_space = neural_env.action_space

#         self.middle_block_size = self.unet.mid_block.in_channels #the out_channels are the same as the in_channels

#         self.mlp = MLP(self.middle_block_size, self.action_space, hidden_dim=256)

#     def forward(self, sparse_encoding):
#         return self.mlp(sparse_encoding)

        
# class QModel(nn.Module):
#     def __init__(self, unet: myUNet2DConditionModel, n_actions:int):
#         super().__init__()
#         self.unet = unet

#         self.middle_block_size = self.unet.mid_block.in_channels #the out_channels are the same as the in_channels

#         self.mlp = MLP(self.middle_block_size, n_actions, hidden_dim=256)

#     def forward(self, sparse_encoding):
#         return self.mlp(sparse_encoding)
        
        
        


# adapted from https://github.com/nicklashansen/tdmpc2/blob/a7890b69857c402ef19edea494e210068e3ec363/tdmpc2/common/layers.py#L85-L123
class NormedLinear(nn.Module): 
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.activation = nn.Mish(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.ln(x)
        return self.activation(x)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.normed_linear1 = NormedLinear(in_dim, hidden_dim)
        self.normed_linear2 = NormedLinear(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x:Tensor):
        # assuming x has shape (batch_size, in_dim, height, width)
        x = x.mean(dim=(-1,-2))
        x = self.normed_linear1(x)
        x = self.normed_linear2(x)
        return self.linear(x)



class AttentionHead(nn.Module):
    def __init__(self, in_dim, out_dim, num_attention_blocks=2, hidden_dim=512):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, 1 + in_dim, in_dim))
        self.attention_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=8,  # number of attention heads
                dim_feedforward=hidden_dim,
                batch_first=True,
                norm_first=True # using layer norm before attention
            )
            for _ in range(num_attention_blocks)
        ])
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).clone()  # Expand cls token for batch
        x = torch.cat((cls_tokens, x), dim=1)  # Prepend cls token
        x += self.position_embedding
        for block in self.attention_blocks:
            x = block(x)
        cls_output = x[:, 0]  # Extract cls token output
        return self.fc(cls_output)
