import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class Expert(nn.Module):
    """
    SwiGLU MLP
    """
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(hidden_size, intermediate_size)
        self.w3 = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # uncomment for it to initially be equivalent to two layer mlp
        # self.w2.weight.data.zero_()
        # self.w2.bias.data.fill_(1.0)
    
    def forward(self, hidden_states):
        hidden_states = F.silu(self.w1(hidden_states)) * self.w2(hidden_states)
        hidden_states = self.layer_norm(self.w3(hidden_states))
        return hidden_states
'''

class Expert(nn.Module):
    """
    SwiGLU MLP
    """
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(hidden_size, intermediate_size)
        self.w3 = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # uncomment for it to initially be equivalent to two layer mlp
        # self.w2.weight.data.zero_()
        # self.w2.bias.data.fill_(1.0)

    def forward(self, hidden_states):
        #hidden_states = F.silu(self.w1(hidden_states)) * self.w2(hidden_states)
        hidden_states = F.relu(self.w1(hidden_states))
        hidden_states = F.relu(self.w2(hidden_states))
        hidden_states = self.w3(hidden_states)
        return hidden_states

class MHRouter(nn.Module):
    def __init__(self, num_experts, hidden_dim, num_heads):
        super().__init__()
        self.expert_embedding = nn.Parameter(torch.randn(hidden_dim // num_heads, num_experts)) # (h, e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : hidden_states (B * L * n, h)
        return torch.matmul(x, self.expert_embedding) # (B * L * n, e)


class MultiHeadMoeBlock(nn.Module):
    def __init__(self, hidden_dim, num_experts, num_heads, topk):
        super().__init__()
        self.topk = topk
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads # h
        self.rounded_dim = (hidden_dim // num_heads) * num_heads # r

        self.multi_head_layer = nn.Linear(hidden_dim, self.rounded_dim)
        self.router = MHRouter(num_experts, hidden_dim, num_heads)

        hidden_size = self.head_dim
        intermediate_size = hidden_size

        self.experts = nn.ModuleList([Expert(hidden_size, intermediate_size) for _ in range(num_experts)])
        self.merge_layer = nn.Linear(self.rounded_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : hidden_states (B, L, d)
        bs, L, _ = x.size()

        # If hidden_dim is not divisible by num_heads r != d
        x = self.multi_head_layer(x) # (B, L, r)
        x = x.reshape(bs * L * self.num_heads, self.head_dim).contiguous() # (B * L * n, h)

        ### Router
        router_logits = self.router(x) # (B * L * n, e)
        router_weights = router_logits.softmax(dim=-1) # (B * L * n, e)
        router_weights, selected_experts = torch.topk(router_weights, self.topk, dim=-1) # (B * L * n, k), (B * L * n, k)
        
        # Call experts densely, faster than selective loops
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # (B * L * n, e, h)
        # Select top-k expert outputs
        selected_expert_outputs = expert_outputs[torch.arange(expert_outputs.size(0)).unsqueeze(1), selected_experts] # (B * L * n, k, h)
        # Multiply selected expert outputs with router weights elementwise
        weighted_expert_outputs = selected_expert_outputs * router_weights.unsqueeze(-1) # (B * L * n, k, h)
        # Combine topk expert outputs
        x = weighted_expert_outputs.sum(dim=1) # (B * L * n, h)
        
        # Back to original shape
        x = x.reshape(bs, L, self.rounded_dim) # (B, L, r)
        x = self.merge_layer(x) # (B, L, d)
        return x
