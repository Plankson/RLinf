import torch
import torch.nn as nn
from torch.nn import functional as F


class MLPResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x

class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_norm1(x) 
        x = self.fc1(x) 
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x) 
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x

class TwinV(nn.Module):
    def __init__(self, input_dim=4096,hidden_dim=256,output_dim=1,num_blocks=5):
        super().__init__()
        self.q1 = MLPResNet(
            num_blocks=num_blocks, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.q2 = MLPResNet(
            num_blocks=num_blocks, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        print("twin Q model size (MB): ", self.get_model_size())

    def get_model_size(self,):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / (1024 * 1024)  # assuming

    def both(self, state,action):
        x = torch.cat([state, action], dim=-1)
        q1_value = self.q1(x)
        q2_value = self.q2(x)
        return q1_value, q2_value

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class SkillHead(nn.Module):
    def __init__(self, input_dim=4096,hidden_dim=256,output_dim=7,num_blocks=5, head_num=2):
        super().__init__()
        self.output_dim = output_dim
        self.phi_list = nn.ModuleList([MLPResNet(
            num_blocks=num_blocks, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim) for _ in range(head_num)])
        print("reward head model size (MB): ", self.get_model_size())

    def get_model_size(self,):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / (1024 * 1024)  # assuming

    def mean_embed(self, obs:torch.Tensor):
        emb_list = [phi(obs) for phi in self.phi_list]
        emb_list = torch.stack(emb_list, dim=-2)
        return emb_list.mean(1)
    
    def get_single_phi(self, obs:torch.Tensor):
        return self.phi_list[0](obs)

    def get_both_phi(self, obs: torch.Tensor):
        phi_list = [phi(obs) for phi in self.phi_list]
        return torch.stack(phi_list, dim=-2)

    def compute_rewards(self, obs: torch.Tensor, next_obs: torch.Tensor, goal: torch.Tensor, type:str='cosine'):
        phi_s =self.get_phi(obs)
        phi_next_s = self.get_phi(next_obs)
        phi_g = self.get_phi(goal)
        z_star = (phi_g - phi_s)/torch.norm(phi_g - phi_s, p=2, dim=-1, keepdim=True)
        reward = torch.sum(F.normalize((phi_next_s - phi_s), p=2, dim=-2) * z_star, dim=-1)
        return reward

    def forward(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor = None,
        return_phi: bool = False,
    ):
        phi_s = self.get_both_phi(obs)
        if return_phi or goal is None:
            return phi_s
        phi_g = self.get_both_phi(goal)
        dist = (phi_s - phi_g).pow(2).sum(dim=-1)
        return -torch.sqrt(dist.clamp(min=1e-6))
