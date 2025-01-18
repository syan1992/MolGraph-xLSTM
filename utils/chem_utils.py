from torch import nn, Tensor

class FGBondEncoder(nn.Module):
    """Encoder of atom."""

    def __init__(self, emb_dim: int):
        """

        Args:
            emb_dim (int): The dimension of the embedding.
        """
        super(FGBondEncoder, self).__init__()

        self.fg_bond_embedding_list = nn.ModuleList()
        fg_bond_embedding_list = [44, 11, 11, 11, 11, 11, 6, 6, 5, 2, 2]

        for i, dim in enumerate(fg_bond_embedding_list):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.fg_bond_embedding_list.append(emb)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Atom embeddings.
        """
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.fg_bond_embedding_list[i](x[:, i])

        return x_embedding

class FGEncoder(nn.Module):
    """Encoder of bond."""

    def __init__(self, emb_dim: int):
        """
        Args:
            emb_dim (int): Dimension of bond embedding.
        """
        super(FGEncoder, self).__init__()

        self.fg_embedding_list = nn.ModuleList()
        full_fg_feature_dims = [11, 6, 6, 6, 6, 2, 2, 11, 8, 8, 8, 2]

        for i, dim in enumerate(full_fg_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.fg_embedding_list.append(emb)

    def forward(self, x: Tensor):
        """
        Args:
            edge_attr (Tensor): Edge attribute.
        """
        fg_embedding = 0
        for i in range(12):#x.shape[1]):
            #print(x[:, i].max())
            #print(x[:, i].min())
            fg_embedding += self.fg_embedding_list[i](x[:, i])

        return fg_embedding

class AtomEncoder(nn.Module):
    """Encoder of atom."""

    def __init__(self, emb_dim: int):
        """

        Args:
            emb_dim (int): The dimension of the embedding.
        """
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = nn.ModuleList()
        full_atom_feature_dims = [44, 11, 11, 11, 11, 11, 6, 6, 5, 2, 2]

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
        full_bond_feature_dims = [4, 2, 6]

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
