import torch
from models.mixture_of_experts import MoE
from models.mhmoe import MultiHeadMoeBlock

class MLP(torch.nn.Module):
    def __init__(self, input_feat, dim_feat, num_tasks, num_layers=3, dropout=0.5, activation=torch.nn.ReLU()):
        super(MLP, self).__init__()

        layers = []


        layers.append(torch.nn.Linear(input_feat, dim_feat))
        layers.append(torch.nn.Dropout(dropout))
        layers.append(activation)
        
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Linear(dim_feat, dim_feat))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(activation)

        layers.append(torch.nn.Linear(dim_feat, num_tasks))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLPMoE(torch.nn.Module):
    def __init__(self, input_feat, dim_feat, num_tasks, num_experts, num_heads):
        super(MLPMoE, self).__init__()
        self.num_heads = num_heads
        self.moe = MoE(dim = input_feat, output_dim = input_feat // self.num_heads, num_experts = num_experts, hidden_dim = dim_feat)

        self.out = torch.nn.ModuleList()
        self.fn = torch.nn.ModuleList()

        for i in range(num_tasks):
            self.out.append(torch.nn.Linear(input_feat, num_tasks))
            self.fn.append(torch.nn.Linear(dim_feat, dim_feat))

        self.query = torch.nn.Linear(dim_feat, dim_feat * self.num_heads)
        self.key = torch.nn.Linear(dim_feat, dim_feat * self.num_heads)
        #freesolv num_experts 8
        self.num_tasks = num_tasks
        self.ln = torch.nn.LayerNorm(dim_feat)
        self.ln_out = torch.nn.LayerNorm(dim_feat // self.num_heads)

        self.dim_feat = dim_feat

        self.mhmoe = MultiHeadMoeBlock(dim_feat, num_experts, self.num_heads, 2)
    def forward(self, x):
        x_input = x.squeeze()
        outputs = []

        x_input_moe = self.query(x_input).view(x_input.shape[0], -1, self.dim_feat)
        x_gate = self.key(x_input).view(x_input.shape[0], -1, self.dim_feat)

        x, loss = self.moe(x_input_moe, x_gate)
        x = x.contiguous().view(x.shape[0], -1)
        outputs.append(self.out[0](x))
        loss_auc = loss

        return outputs[0], loss_auc
