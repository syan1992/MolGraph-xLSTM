from typing import Any, List

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, Set2Set
from torch_geometric.utils import to_dense_batch

from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        FeedForwardConfig,
    )

from models.deepgcn_vertex import GENConv
from models.deepgcn_nn import MLP, norm_layer
from utils.chem_uti import AtomEncoder, BondEncoder, FGEncoder

class xLSTMModule(torch.nn.Module):
    def __init__(self, num_blocks: int, embedding_dim: int, slstm: List[int], proj_factor: int = 2, act_fn: str = "relu"):
        super().__init__()

        cfg = xLSTMBlockStackConfig(
            mlstm_block = mLSTMBlockConfig(
                mlstm = mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block = sLSTMBlockConfig(
                slstm = sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=proj_factor, act_fn=act_fn)
            ),
            context_length = 512,
            num_blocks = num_blocks,
            embedding_dim = embedding_dim,
            slstm_at = slstm,
        )

        self.xlstm = xLSTMBlockStack(cfg)

    def forward(self, f_node):
        h_lstm = self.xlstm(f_node)
        return h_lstm

class DeeperGCN(torch.nn.Module):
    """DeeperGCN network."""

    def __init__(
        self,
        num_gc_layers: int,
        dropout: float,
        aggr: str = "add",
        mlp_layers: int = 1,
        power: int = 4,
        dims: int = 64,
    ):
        """
        Args:
            num_gc_layers (int): Depth of the network.
            dropout (float): Dropout rate.
            aggr (str, optional): Selection of aggregation methods. add, sum or max. Defaults to "add".
            mlp_layers (int, optional): Number of MLP layers. Defaults to 1.
            power (int, optional): Number of layers used in the output.
            dims (int, optional): Number of dimensions. Defaults to 64.
        """
        super(DeeperGCN, self).__init__()

        self.powers = list(range(power))
        self.num_gc_layers = num_gc_layers
        self.dropout = dropout
        aggr = aggr

        t = 0.1
        self.learn_t = False
        p = 1.0
        self.learn_p = False
        y = 0.0
        self.learn_y = False
        self.msg_norm = False
        learn_msg_scale = False
        mlp_layers = mlp_layers
        norm = "layer"

        self.ffn = torch.nn.ModuleList()
        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.virtualnode_embedding = torch.nn.Embedding(1, dims * power)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for _ in range(num_gc_layers - 1):
            self.mlp_virtualnode_list.append(MLP([dims * power] * 3, norm=norm))

        for i in range(num_gc_layers):
            conv = GENConv(
                dims * power,
                dims * power,
                aggr = aggr,
                t = t,
                learn_t = self.learn_t,
                p = p,
                learn_p = self.learn_p,
                y = y,
                learn_y = self.learn_p,
                msg_norm = self.msg_norm,
                learn_msg_scale = learn_msg_scale,
                encode_edge = False,
                bond_encoder = True,
                norm = norm,
                mlp_layers = mlp_layers,
            )
            self.gcns.append(conv)
            self.norms.append(norm_layer(norm, dims * power))
            self.ffn.append(torch.nn.Linear(dims * power, dims))

        self.atom_encoder = AtomEncoder(emb_dim=dims * power)
        self.bond_encoder = BondEncoder(emb_dim=dims * power)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_attr = graph_batch.edge_attr
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        h = self.atom_encoder(x)
        h_init = h
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )
        h = h + virtualnode_embedding[batch]
        edge_emb = self.bond_encoder(edge_attr)
        out = []
        h = self.gcns[0](h, edge_index, edge_emb)
        out.append(self.ffn[0](h))
        for layer in range(1, self.num_gc_layers):
            h1 = self.norms[layer - 1](h)
            h2 = F.relu(h1)
            virtualnode_embedding_temp = global_add_pool(h2, batch) + virtualnode_embedding
            virtualnode_embedding = F.dropout(
                self.mlp_virtualnode_list[layer - 1](virtualnode_embedding_temp),
                self.dropout,
                training=self.training,
            )
            h2 = h2 + virtualnode_embedding[batch]
            h_res = h
            h = self.gcns[layer](h2, edge_index, edge_emb)
            h = h + h_res
            out.append(self.ffn[layer](h))
        h_graph = torch.cat([out[p] for p in self.powers], dim=-1)
        return h_graph, h_init

class GraphxLSTM(torch.nn.Module):
    def __init__(self, opt: Any):
        super().__init__()

        self.classification = opt.classification

        #GCN
        num_gc_layers = opt.num_gc_layers
        dropout = 0.2
        aggr = "add"
        mlp_layers = opt.mlp_layers

        self.atom_graph_gnn = DeeperGCN(
            num_gc_layers,
            dropout,
            aggr,
            mlp_layers = mlp_layers,
            power = opt.power,
            dims = opt.num_dim,
        )

        #xLSTM
        if opt.classification:
            act_fn = 'relu'
        else:
            act_fn = 'gelu'

        self.atom_lstm = xLSTMModule(
            num_blocks = opt.num_blocks,
            embedding_dim = opt.power * opt.num_dim,
            slstm = opt.slstm,
            act_fn = act_fn
        )

        self.fg_lstm = xLSTMModule(
            num_blocks=opt.num_blocks,
            embedding_dim=opt.power * opt.num_dim,
            slstm=opt.slstm,
            act_fn = act_fn
        )

        self.fg_encoder = FGEncoder(emb_dim = opt.power * opt.num_dim)

        self.reg = torch.nn.Linear(opt.power * opt.num_dim, opt.power * opt.num_dim)
        self.cls = torch.nn.Linear(opt.power * opt.num_dim * 2, opt.power * opt.num_dim)
        self.set2set = Set2Set(opt.power * opt.num_dim, processing_steps=3)

    def forward(self, batch: Tensor):
        #GCN
        h_graph, _ = self.atom_graph_gnn(batch)

        #atom-level xLSTM
        atom_batch = batch.batch
        f_node, mask = to_dense_batch(h_graph, atom_batch)
        h_atom_lstm = self.atom_lstm(f_node)

        #fg-level xLSTM
        fg_x = self.fg_encoder(batch.fg_x.long())
        fg_batch = batch.fg_x_batch
        f_node_fg, mask_fg = to_dense_batch(fg_x, fg_batch)
        h_fg_lstm = self.fg_lstm(f_node_fg)

        if self.classification:
            h_graph = self.cls(self.set2set(h_graph, atom_batch))
            h_lstm = (global_max_pool(h_atom_lstm[mask], atom_batch))
            fg_lstm = (global_max_pool(h_fg_lstm[mask_fg], fg_batch))

            h_out = h_lstm + fg_lstm + h_graph
        else:
            h_graph = self.reg(global_add_pool(h_graph, atom_batch))
            h_lstm = global_max_pool(h_atom_lstm[mask], atom_batch)
            fg_lstm = global_max_pool(h_fg_lstm[mask_fg], fg_batch)

            h_out = h_lstm + fg_lstm + h_graph

        return h_out, h_lstm, fg_lstm, h_graph
