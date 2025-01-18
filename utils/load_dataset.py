import os
import re
import random
from itertools import repeat
from typing import Callable

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import AddRandomWalkPE

from tqdm import tqdm
from collections import deque
import pandas as pd
import numpy as np

from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Avalon import pyAvalonTools
from unimol_tools import UniMolRepr
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from ogb.utils.mol import smiles2graph
from transformers import RobertaTokenizerFast

from utils.chem import mol_to_graphs

FINGERPRINT_SIZE = 2048

def dfs_torch_geometric(edge_index, start_node=0):
    """Deep First Search"""
    nodes = set(edge_index[0].tolist()) | set(edge_index[1].tolist())

    visited = set()
    order = []

    stack = deque([start_node])

    while stack:
        current_node = stack.pop()
        if current_node not in visited:
            visited.add(current_node)
            order.append(current_node)

            neighbors = edge_index[1, edge_index[0] == current_node].tolist() + \
                        edge_index[0, edge_index[1] == current_node].tolist()

            for neighbor in reversed(sorted(neighbors)):
                if neighbor not in visited:
                    stack.append(neighbor)

    for node in nodes:
        if node not in visited:
            stack = deque([node])
            while stack:
                current_node = stack.pop()
                if current_node not in visited:
                    visited.add(current_node)
                    order.append(current_node)

                    neighbors = edge_index[1, edge_index[0] == current_node].tolist() + \
                                edge_index[0, edge_index[1] == current_node].tolist()

                    for neighbor in reversed(sorted(neighbors)):
                        if neighbor not in visited:
                            stack.append(neighbor)
    return order


def bfs_torch_geometric(edge_index, start_node=0):
    """Breadth First Search"""
    nodes = set(edge_index[0].tolist()) | set(edge_index[1].tolist())

    visited = set()
    order = []

    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            order.append(current_node)

            neighbors = edge_index[1, edge_index[0] == current_node].tolist() + \
                        edge_index[0, edge_index[1] == current_node].tolist()

            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

    for node in nodes:
        if node not in visited:
            queue = deque([node])
            while queue:
                current_node = queue.popleft()
                if current_node not in visited:
                    visited.add(current_node)
                    order.append(current_node)

                    neighbors = edge_index[1, edge_index[0] == current_node].tolist() + \
                                edge_index[0, edge_index[1] == current_node].tolist()

                    for neighbor in neighbors:
                        if neighbor not in visited:
                            queue.append(neighbor)

    return order

def getmorganfingerprint(mol: Mol):
    """Get the ECCP fingerprint.

    Args:
        mol (Mol): The molecule.
    """
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FINGERPRINT_SIZE))


def getmaccsfingerprint(mol: Mol):
    """Get the MACCS fingerprint.

    Args:
        mol (Mol): The molecule.
    """
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

class PygOurDataset(InMemoryDataset):
    """Load datasets."""

    def __init__(
        self,
        root: str = "dataset",
        phase: str = "train",
        dataname: str = "hiv",
        smiles2graph: Callable = smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root (str, optional): The local position of the dataset. Defaults to "dataset".
            phase (str, optional): The data is train, validation or test set. Defaults to "train".
            dataname (str, optional): The name of the dataset. Defaults to "hiv".
            smiles2graph (Callable, optional): Generate the molecular graph from the SMILES
                string. Defaults to smiles2graph.
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = os.path.join(root, dataname)
        self.version = 1
        self.dataname = dataname
        self.phase = phase
        self.aug = "none"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            "seyonec/ChemBERTa_zinc250k_v2_40k", max_len=100
        )

        self.tokenizer_simple = SmilesTokenizer('uti/vocab.txt')
        self.transform_pe = AddRandomWalkPE(walk_length=5) 
        self.geom3d = UniMolRepr(data_type='molecule')
        super(PygOurDataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        """Return the name of the raw file."""
        return self.phase + "_" + self.dataname + ".csv"

    @property
    def processed_file_names(self):
        """Return the name of the processed file."""
        return self.phase + "_" + self.dataname + ".pt"

    def process(self):
        """Generate the processed file from the raw file. Only execute when the data is loaded firstly."""
        data_df = pd.read_csv(
            os.path.join(self.raw_dir, self.phase + "_" + self.dataname + ".csv")
        )
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df[data_df.columns.difference(["smiles", "mol_id", "num", "name"])]

        max_len = 100
        tokenized_smiles = [self.tokenizer_simple.encode(smiles) for smiles in smiles_list]
        truncated_padded_sequences = [
                seq[:max_len] + [0] * (max_len - len(seq))
                if len(seq) < max_len else seq[:max_len]
                for seq in tokenized_smiles
        ]

        encodings = self.tokenizer(smiles_list.tolist(), truncation=True, padding=True)

        print("Converting SMILES strings into graphs...")
        data_list = []

        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list.iloc[i]
            graph = self.smiles2graph(smiles)

            sorted_order_b = bfs_torch_geometric(graph['edge_index'])
            sorted_order_d = dfs_torch_geometric(graph['edge_index'])

            if len(sorted_order_b) != len(graph["node_feat"]):
                for k in range(len(graph["node_feat"])):
                    if k not in sorted_order_b:
                        sorted_order_b.append(k)

            if len(sorted_order_d) != len(graph["node_feat"]):
                for k in range(len(graph["node_feat"])):
                    if k not in sorted_order_d:
                        sorted_order_d.append(k)

            fgs, clusters, atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list = mol_to_graphs(smiles)

            atom2fg_list = sorted(atom2fg_list, key=lambda x: x[0])
            rdkit_mol = AllChem.MolFromSmiles(smiles)

            mgf = getmorganfingerprint(rdkit_mol)
            maccs = getmaccsfingerprint(rdkit_mol)
            avalon = pyAvalonTools.GetAvalonFP(rdkit_mol)
            avalon = [int(b) for b in avalon.ToBitString()]

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = len(fg_features) #int(graph["num_nodes"])
            if len(fg_edge_list) > 0:
                fg_edge_index = np.vstack(fg_edge_list).T
            else:
                fg_edge_index = np.empty((2, 0), dtype=int)

            sorted_order_b_fg = bfs_torch_geometric(fg_edge_index)
            sorted_order_d_fg = dfs_torch_geometric(fg_edge_index)

            if len(sorted_order_b_fg) != len(fg_features):
                for k in range(len(fg_features)):
                    if k not in sorted_order_b_fg:
                        sorted_order_b_fg.append(k)

            if len(sorted_order_d_fg) != len(fg_features):
                for k in range(len(graph["node_feat"])):
                    if k not in sorted_order_d_fg:
                        sorted_order_d_fg.append(k)
            
            if len(fg_features) == 0:
                fg_features = [[0] * 12]
            
            data.edge_index = torch.Tensor(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.Tensor(bond_features).to(torch.int64)
            data.x = torch.Tensor(atom_features).to(torch.int64)
            data.y = torch.Tensor([homolumogap])
            data.input_ids = torch.Tensor(encodings.input_ids[i])
            data.attention_mask = torch.Tensor(encodings.attention_mask[i])
            data.token_ids = torch.Tensor(truncated_padded_sequences[i])
            data.sorted_order_b = torch.Tensor(sorted_order_b).to(torch.int64)
            data.sorted_order_d = torch.Tensor(sorted_order_d).to(torch.int64)
            data.sorted_order_b_fg = torch.Tensor(sorted_order_b_fg).to(torch.int64)
            data.sorted_order_d_fg = torch.Tensor(sorted_order_d_fg).to(torch.int64)
            data.atom2fg_list = torch.Tensor(atom2fg_list).to(torch.int64)
            data.clusters = torch.Tensor(clusters).to(torch.int64)
            data.fg_x = torch.Tensor(fg_features)
            data.fg_edge = torch.Tensor(fg_edge_list).to(torch.int64)
            data.fg_edge_attr = torch.Tensor(fg_edge_features).to(torch.int64)
            data.mgf = torch.tensor(mgf)
            data.maccs = torch.tensor(maccs)
            data.avalon = torch.tensor(avalon)
            data.smiles = smiles
            #data.geom3d_feature = torch.tensor(unimol_feature[smiles])#torch.tensor(self.geom3d.get_repr(smiles)).squeeze()[0]
            data_list.append(data)

        data_list = [self.transform_pe(data) for data in data_list]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx: int):
        """Get the idx-th data.
        Args:
            idx (int): The number of the data.
        """
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            if key=='smiles':
                    data[key] = item[idx]
            else:
                    s = list(repeat(slice(None), item.dim()))
                    s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                    data[key] = item[s]
        random_integer = random.randint(0, len(data.fg_x)-1)

        ''''
        sorted_order_b_fg = dfs_torch_geometric(data.fg_edge.T, start_node = random_integer)
        if len(sorted_order_b_fg) != len(data.fg_x):
            for i in range(len(data.fg_x)):
                if i not in sorted_order_b_fg:
                    sorted_order_b_fg.append(i)
        data.sorted_order_b_fg = torch.Tensor(sorted_order_b_fg).to(torch.int64)

        random_integer = random.randint(0, len(data.x) - 1)
        sorted_order_b = dfs_torch_geometric(data.edge_index, start_node = random_integer)
        if len(sorted_order_b) != len(data.x):
            for i in range(len(data.x)):
                if i not in sorted_order_b:
                    sorted_order_b.append(i)
        data.sorted_order_b = torch.Tensor(sorted_order_b).to(torch.int64)
        '''
        return data
