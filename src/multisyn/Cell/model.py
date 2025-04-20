import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv

class GATEncoder(nn.Module):
    """Use multi-layer GATConv to extract graph structure node features"""
    def __init__(self, in_features: int, out_features: int, num_heads: int = 1, activation=F.relu, k: int = 2):
        super(GATEncoder, self).__init__()
        assert k >= 2
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.conv = nn.ModuleList()
        self.conv.append(GATConv(in_features, 2 * out_features, num_heads=num_heads))
        self.activation = activation
        for _ in range(1, k - 1):
            self.conv.append(GATConv(2 * out_features * num_heads, 2 * out_features, num_heads=num_heads))
        self.conv.append(GATConv(2 * out_features * num_heads, out_features, num_heads=1))

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        """Input graph and node features, return encoded node embedding."""
        for conv in self.conv:
            x = self.activation(conv(g, x).flatten(1))
        return x

class Cell2Vec(nn.Module):
    """Cell lineage state modeling using graph neural network based on PPI graph"""
    def __init__(self, encoder: GATEncoder, n_cell, n_dim):
        super(Cell2Vec, self).__init__()
        self.encoder = encoder
        self.embeddings = nn.Embedding(n_cell, n_dim)
        self.projector = nn.Sequential(
            nn.Linear(encoder.out_features, n_dim),
            nn.Dropout()
        )
        # 实现公式（2）中的f(Z)操作，将 encoder 输出的特征维度投影到 n_dim，然后是一个 nn.Dropout 层，帮助防止模型过拟合。    
    def forward(self, g: dgl.DGLGraph, x: torch.Tensor,
                x_indices: torch.LongTensor, c_indices: torch.LongTensor):
        encoded = self.encoder(g, x) # ppi
        encoded = encoded.index_select(0, x_indices)    # 从 encoded 中选择与基因索引 x_indices 对应的节点特征。这样可以确保只选择特定的节点特征。
        proj = self.projector(encoded).permute(1, 0)    # permute(1, 0) 是为了改变张量的维度顺序，使其与细胞嵌入相乘时的维度匹配，PPI ⽹络中获取的基因特征矩阵
        emb = self.embeddings(c_indices)    # c_j 细胞系的状态特征
        out = torch.mm(emb, proj)   # 实现公式（2）o^t_j = f(Z) · c_j,
        # 基因隐藏状态和细胞系状态的交互，产生了基因的显性状态向量，表示在当前细胞系中的基因表现
        return out


class RandomW(nn.Module):

    def __init__(self, n_node, n_node_dim, n_cell, n_dim):
        super(RandomW, self).__init__()
        self.encoder = nn.Embedding(n_node, n_node_dim)
        self.embeddings = nn.Embedding(n_cell, n_dim)
        self.projector = nn.Sequential(
            nn.Linear(n_node_dim, n_dim),
            nn.Dropout()
        )

    def forward(self, x_indices: torch.LongTensor, c_indices: torch.LongTensor):
        encoded = self.encoder(x_indices)
        proj = self.projector(encoded).permute(1, 0)
        emb = self.embeddings(c_indices)
        out = torch.mm(emb, proj)
        return out
