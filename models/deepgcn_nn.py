from torch import nn, Tensor
from torch.nn import Sequential as Seq, Linear as Lin
import pdb

allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


def get_atom_feature_dims():
    """Features of atom."""
    return list(
        map(
            len,
            [
                allowable_features["possible_atomic_num_list"],
                allowable_features["possible_chirality_list"],
                allowable_features["possible_degree_list"],
                allowable_features["possible_formal_charge_list"],
                allowable_features["possible_numH_list"],
                allowable_features["possible_number_radical_e_list"],
                allowable_features["possible_hybridization_list"],
                allowable_features["possible_is_aromatic_list"],
                allowable_features["possible_is_in_ring_list"],
            ],
        )
    )


def get_bond_feature_dims():
    """Features of bond."""
    return list(
        map(
            len,
            [
                allowable_features["possible_bond_type_list"],
                allowable_features["possible_bond_stereo_list"],
                allowable_features["possible_is_conjugated_list"],
            ],
        )
    )


##############################
#    Basic layers
##############################
def act_layer(act_type: str, inplace: bool = False, neg_slope: float = 0.2, n_prelu: int = 1):
    """Activation layer.

    Args:
        act_type (str): Type of activation
        inplace (bool, optional): The parameter for ReLU and LeakyReLU. Defaults to False.
        neg_slope (float, optional): The parameter for LeakyReLU . Defaults to 0.2.
        n_prelu (int, optional): The parameter for PReLU . Defaults to 1.
    """

    act = act_type.lower()
    if act == "relu":
        layer = nn.ReLU(inplace)
    elif act == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError("activation layer [%s] is not found" % act)
    return layer


def norm_layer(norm_type: str, nc: int):
    """Normalization Layer.

    Args:
        norm_type (str): Type of the normalization
        nc (int): Number of features
    """
    # normalization layer 1d
    norm = norm_type.lower()
    if norm == "batch":
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == "layer":
        layer = nn.LayerNorm(nc, elementwise_affine=True)
    elif norm == "instance":
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm)
    return layer


class MLP(Seq):
    """Multi-layer perceptron."""

    def __init__(
        self,
        channels: int,
        act: str = "relu",
        norm: str = None,
        bias: bool = True,
        drop: float = 0.0,
        last_lin: bool = False,
    ):
        m = []

        for i in range(1, len(channels)):

            m.append(Lin(channels[i - 1], channels[i], bias))

            if (i == len(channels) - 1) and last_lin:
                pass
            else:
                if norm is not None and norm.lower() != "none":
                    m.append(norm_layer(norm, channels[i]))
                if act is not None and act.lower() != "none":
                    m.append(act_layer(act))
                if drop > 0:
                    m.append(nn.Dropout2d(drop))

        self.m = m
        super(MLP, self).__init__(*self.m)


class AtomEncoder(nn.Module):
    """Encoder of atom."""

    def __init__(self, emb_dim: int):
        """

        Args:
            emb_dim (int): The dimension of the embedding.
        """
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = nn.ModuleList()
        full_atom_feature_dims = get_atom_feature_dims()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Atom embeddings.
        """
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(nn.Module):
    """Encoder of bond."""

    def __init__(self, emb_dim: int):
        """
        Args:
            emb_dim (int): Dimension of bond embedding.
        """
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = nn.ModuleList()
        full_bond_feature_dims = get_bond_feature_dims()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr: Tensor):
        """
        Args:
            edge_attr (Tensor): Edge attribute.
        """
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding
