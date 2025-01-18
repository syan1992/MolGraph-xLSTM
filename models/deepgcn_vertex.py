import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, scatter_softmax
from torch_geometric.utils import degree

from models.deepgcn_nn import MLP, BondEncoder
import pdb

class GenMessagePassing(MessagePassing):
    """Aggregation methods from DeeperGCN."""

    def __init__(
        self, aggr="softmax", t=1.0, learn_t=False, p=1.0, learn_p=False, y=0.0, learn_y=False
    ):

        if aggr in ["softmax_sg", "softmax", "softmax_sum"]:

            super(GenMessagePassing, self).__init__(aggr=None)
            self.aggr = aggr

            if learn_t and (aggr == "softmax" or aggr == "softmax_sum"):
                self.learn_t = True
                self.t = torch.nn.Parameter(torch.Tensor([t]), requires_grad=True)
            else:
                self.learn_t = False
                self.t = t

            if aggr == "softmax_sum":
                self.y = torch.nn.Parameter(torch.Tensor([y]), requires_grad=learn_y)

        elif aggr in ["power", "power_sum"]:

            super(GenMessagePassing, self).__init__(aggr=None)
            self.aggr = aggr

            if learn_p:
                self.p = torch.nn.Parameter(torch.Tensor([p]), requires_grad=True)
            else:
                self.p = p

            if aggr == "power_sum":
                self.y = torch.nn.Parameter(torch.Tensor([y]), requires_grad=learn_y)
        else:
            super(GenMessagePassing, self).__init__(aggr=aggr)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):

        if self.aggr in ["add", "mean", "max", None]:
            return super(GenMessagePassing, self).aggregate(inputs, index, ptr, dim_size)

        elif self.aggr in ["softmax_sg", "softmax", "softmax_sum"]:

            if self.learn_t:
                out = scatter_softmax(inputs * self.t, index, dim=self.node_dim)
            else:
                with torch.no_grad():
                    out = scatter_softmax(inputs * self.t, index, dim=self.node_dim)

            out = scatter(inputs * out, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")

            if self.aggr == "softmax_sum":
                self.sigmoid_y = torch.sigmoid(self.y)
                degrees = degree(index, num_nodes=dim_size).unsqueeze(1)
                out = torch.pow(degrees, self.sigmoid_y) * out

            return out

        elif self.aggr in ["power", "power_sum"]:
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(
                torch.pow(inputs, self.p),
                index,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce="mean",
            )
            torch.clamp_(out, min_value, max_value)
            out = torch.pow(out, 1 / self.p)

            if self.aggr == "power_sum":
                self.sigmoid_y = torch.sigmoid(self.y)
                degrees = degree(index, num_nodes=dim_size).unsqueeze(1)
                out = torch.pow(degrees, self.sigmoid_y) * out

            return out

        else:
            raise NotImplementedError("To be implemented")


class MsgNorm(torch.nn.Module):
    """The message normalization layer proposed by DeeperGCN."""

    def __init__(self, learn_msg_scale=False):
        super(MsgNorm, self).__init__()

        self.msg_scale = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=learn_msg_scale)

    def forward(self, x, msg, p=2):
        msg = F.normalize(msg, p=p, dim=1)
        x_norm = x.norm(p=p, dim=1, keepdim=True)
        msg = msg * x_norm * self.msg_scale
        return msg


class GENConv(GenMessagePassing):
    """
    GENeralized Graph Convolution (GENConv): https://arxiv.org/pdf/2006.07739.pdf
    SoftMax & PowerMean Aggregation.
    """

    def __init__(
        self,
        in_dim,
        emb_dim,
        aggr="softmax",
        t=1.0,
        learn_t=False,
        p=1.0,
        learn_p=False,
        y=0.0,
        learn_y=False,
        msg_norm=False,
        learn_msg_scale=True,
        encode_edge=False,
        bond_encoder=False,
        edge_feat_dim=None,
        norm="batch",
        mlp_layers=2,
        eps=1e-7,
    ):

        super(GENConv, self).__init__(
            aggr=aggr, t=t, learn_t=learn_t, p=p, learn_p=learn_p, y=y, learn_y=learn_y
        )

        channels_list = [in_dim]

        for i in range(mlp_layers - 1):
            channels_list.append(in_dim * 2)

        channels_list.append(emb_dim)

        self.mlp = MLP(channels=channels_list, norm=norm, last_lin=True)

        self.msg_encoder = torch.nn.ReLU()
        self.eps = eps

        self.msg_norm = msg_norm
        self.encode_edge = encode_edge
        self.bond_encoder = bond_encoder

        if msg_norm:
            self.msg_norm = MsgNorm(learn_msg_scale=learn_msg_scale)
        else:
            self.msg_norm = None

        if self.encode_edge:
            if self.bond_encoder:
                self.edge_encoder = BondEncoder(emb_dim=in_dim)
            else:
                self.edge_encoder = torch.nn.Linear(in_dim, in_dim)

    def forward(self, x, edge_index, edge_attr=None):
        if self.encode_edge and edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = edge_attr

        m = self.propagate(edge_index, x=x, edge_attr=edge_emb)

        if self.msg_norm is not None:
            m = self.msg_norm(x, m)

        h = x + m

        out = self.mlp(h)

        return out

    def message(self, x_j, edge_attr=None):

        if edge_attr is not None:
            msg = x_j + edge_attr
        else:
            msg = x_j

        return self.msg_encoder(msg) + self.eps

    def update(self, aggr_out):
        return aggr_out


class GINEConv(tg.nn.GINEConv):
    """GINConv layer (with activation, batch normalization)"""

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True, aggr="add"):
        super(GINEConv, self).__init__(MLP([in_channels, out_channels], act, norm, bias))

    def forward(self, x, edge_index, edge_attr):
        return super(GINEConv, self).forward(x, edge_index, edge_attr)
