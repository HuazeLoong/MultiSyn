import dgl
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_func
from dgl import function as fn
from functools import partial
from torch.nn import Sequential, Linear, ReLU,BatchNorm1d

# dgl graph utils
def reverse_edge(tensor):
    n = tensor.size(0)  
    assert n % 2 == 0  
    delta = torch.ones(n).type(torch.long) 
    delta[torch.arange(1, n, 2)] = -1  
    return tensor[delta + torch.tensor(range(n))] 

def del_reverse_message(edge, field):
    return {'m': edge.src[field] - edge.data['rev_h']}  

def add_attn(node, field, attn):
    feat = node.data[field].unsqueeze(1)  
    return {
        field: (attn(feat, node.mailbox['m'], node.mailbox['m']) + feat).squeeze(1)  # 残差连接
    }  

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'" 
    d_k = query.size(-1) # 获取查询（query）最后一个向量的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)     # 计算注意力分数
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()  
        assert d_model % h == 0  # 维度可被头数整除
        # We assume d_v always equals d_k
        self.d_k = d_model // h 
        self.h = h  

        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 存储注意力权重
        self.dropout = nn.Dropout(p=dropout)  

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"  
        if mask is not None:
            mask = mask.unsqueeze(1)  
        nbatches = query.size(0)  
        """
        对查询、键和值进行线性映射，并按头的数量进行拆分和重新排列维度
        将经过线性变换后的张量重新调整形状，并对维度进行转置操作"""
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # 计算多头注意力
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        # 重新组合多头注意力的结果，并通过线性变换输出
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Node_GRU(nn.Module):
    """GRU for graph readout. Implemented with dgl graph"""
    def __init__(self, hid_dim, bidirectional=True):
        super(Node_GRU, self).__init__()
        self.hid_dim = hid_dim
        # 设置GRU层是否为双向的
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.att_mix = MultiHeadedAttention(6, hid_dim)
        self.gru = nn.GRU(hid_dim, hid_dim, batch_first=True, bidirectional=bidirectional)
    
    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]
        node_size = bg.batch_num_nodes(ntype)
        start_index = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_size, 0)[:-1]])

        max_num_node = max(node_size)

        hidden_lst = []
        for i in range(bg.batch_size):
            start, size = start_index[i], node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)
            cur_hidden = torch.nn.ZeroPad2d((0, 0, 0, max_num_node - cur_hidden.shape[0]))(cur_hidden)

            hidden_lst.append(cur_hidden.unsqueeze(0))
        hidden_lst = torch.cat(hidden_lst, 0) # 行方向拼接
        return hidden_lst
        
    def forward(self, bg, suffix='h'):
        """
        bg: dgl.Graph (batch)
        hidden states of nodes are supposed to be in field 'h'.
        """
        self.suffix = suffix
        device = bg.device
        
        p_pharmj = self.split_batch(bg, 'p', f'f_{suffix}', device)
        a_pharmj = self.split_batch(bg, 'a', f'f_{suffix}', device)

        # 生成掩码，用于屏蔽无效的节点特征
        mask = (a_pharmj != 0).type(torch.float32).matmul((p_pharmj.transpose(-1, -2) != 0).
                                                          type(torch.float32)) == 0

        # 多头注意力机制，用于融合节点特征
        h = self.att_mix(a_pharmj, p_pharmj, p_pharmj, mask) + a_pharmj

        # 将隐状态扩展为与GRU层相匹配的维度
        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction, 1, 1)
        h, hidden = self.gru(h, hidden)
        
        # 取平均值并减少节点特征的维度
        graph_embed = []
        node_size = bg.batch_num_nodes('p')
        start_index = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_size, 0)[:-1]])

        for i in range(bg.batch_size):
            start, size = start_index[i], node_size[i]
            graph_embed.append(h[i, :size].view(-1, self.direction * self.hid_dim).mean(0).unsqueeze(0))

        graph_embed = torch.cat(graph_embed, 0)

        return graph_embed

       
class MVMP(nn.Module):
    def __init__(self, msg_func=add_attn, hid_dim=300, depth=3, view='aba', suffix='h', act=nn.ReLU()):
        """
        MultiViewMassagePassing
        view: a：single, ap：, apj
        suffix: filed to save the nodes' hidden state in dgl.graph. 
                e.g. bg.nodes[ntype].data['f'+'_junc'(in ajp view)+suffix]
        """
        super(MVMP, self).__init__()
        self.view = view  
        self.depth = depth  
        self.suffix = suffix  # 节点隐藏状态的后缀，用于在dgl.graph中保存节点的隐藏状态
        self.msg_func = msg_func  
        self.act = act  
        self.homo_etypes = [('a', 'b', 'a')] 
        self.hetero_etypes = []  
        self.node_types = ['a', 'p']  
        if 'p' in view: 
            self.homo_etypes.append(('p', 'r', 'p'))  
        if 'j' in view:
            self.node_types.append('junc')  
            self.hetero_etypes = [('a', 'j', 'p'), ('p', 'j', 'a')]  

        self.attn = nn.ModuleDict()
        for etype in self.homo_etypes + self.hetero_etypes:
            self.attn[''.join(etype)] = MultiHeadedAttention(4, hid_dim)

        # 多层消息传递
        self.mp_list = nn.ModuleDict()
        for edge_type in self.homo_etypes: # 同构边类型
            self.mp_list[''.join(edge_type)] = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for i in range(depth - 1)])

        # 节点特征更新层
        self.node_last_layer = nn.ModuleDict()
        for ntype in self.node_types:
            self.node_last_layer[ntype] = nn.Linear(3 * hid_dim, hid_dim)

    def update_edge(self, edge, layer):
        return {'h': self.act(edge.data['x'] + layer(edge.data['m']))}
    
    def update_node(self, node, field, layer):
        return {field: layer(torch.cat([node.mailbox['mail'].sum(dim=1), node.data[field], node.data['f']], 1))}
    """ 
        mailbox['mail'] 包含每个节点接收到的来自邻居节点的消息；
        data[field] 表示节点当前的特征；
        data['f'] 表示节点的全局特征。
    """

    def init_node(self, node):
        return {f'f_{self.suffix}': node.data['f'].clone()} # 为什么要对其克隆？

    def init_edge(self, edge):
        return {'h': edge.data['x'].clone()}          # 为什么要对其克隆？

    def forward(self, bg):
        suffix = self.suffix
        # 初始化节点特征和边特征
        for ntype in self.node_types:
            if ntype != 'junc':
                bg.apply_nodes(self.init_node, ntype=ntype)
        for etype in self.homo_etypes:
            bg.apply_edges(self.init_edge, etype=etype)

        if 'j' in self.view:
            bg.nodes['a'].data[f'f_junc_{suffix}'] = bg.nodes['a'].data['f_junc'].clone()
            bg.nodes['p'].data[f'f_junc_{suffix}'] = bg.nodes['p'].data['f_junc'].clone()

        # 构建消息传递函数
        update_funcs = {e: (fn.copy_e('h', 'm'), partial(self.msg_func, attn=self.attn[''.join(e)],
                                                          field=f'f_{suffix}')) for e in self.homo_etypes}
        update_funcs.update({e: (fn.copy_src(f'f_junc_{suffix}', 'm'), partial(self.msg_func, 
                        attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')) for e in self.hetero_etypes})

        # 消息传递过程 message passing
        for i in range(self.depth - 1):  # 第一次消息传递已经在初始化阶段完成
            bg.multi_update_all(update_funcs, cross_reducer='sum') # 对节点进行消息传递       疑问？
            for edge_type in self.homo_etypes: # 遍历同质边
                bg.edges[edge_type].data['rev_h'] = reverse_edge(bg.edges[edge_type].data['h'])
                bg.apply_edges(partial(del_reverse_message, field=f'f_{suffix}'), etype=edge_type)
                bg.apply_edges(partial(self.update_edge, layer=self.mp_list[''.join(edge_type)][i]), 
                               etype=edge_type)

        # 更新节点特征 last update of node feature
        update_funcs = {e: (fn.copy_e('h', 'mail'), partial(self.update_node, 
                        field=f'f_{suffix}', layer=self.node_last_layer[e[0]])) for e in self.homo_etypes}
        bg.multi_update_all(update_funcs, cross_reducer='sum')

        # 更新连接节点特征 last update of junc feature
        bg.multi_update_all({e: (fn.copy_src(f'f_junc_{suffix}', 'mail'), 
                partial(self.update_node, field=f'f_junc_{suffix}', 
                layer=self.node_last_layer['junc'])) for e in self.hetero_etypes}, 
                cross_reducer='sum')

