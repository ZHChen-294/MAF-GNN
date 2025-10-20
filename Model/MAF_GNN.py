import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from einops import rearrange



class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.sparsity = 30
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def _collate_adjacency(self, a):
        i_list, v_list = [], []

        for sample, _a in enumerate(a):
            thresholded_a = (_a > np.percentile(_a.detach().cpu().numpy(), 100 - self.sparsity))
            _i = thresholded_a.nonzero(as_tuple=False)
            _v = torch.ones(len(_i))
            _i += sample * a.shape[1]
            i_list.append(_i)
            v_list.append(_v)

        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        return torch.sparse_coo_tensor(_i, _v,
                                       (a.shape[0] * a.shape[1],
                                        a.shape[0] * a.shape[1])).to_sparse_csr()

    def forward(self, x, adj):
        batch_size, num_nodes = x.shape[:2]
        x = rearrange(x, 'b n c -> (b n) c')
        adj = self._collate_adjacency(adj)

        x = F.dropout(F.relu(self.conv1(x, adj)), p=0.1, training=self.training)
        x = F.dropout(F.relu(self.conv2(x, adj)), p=0.1, training=self.training)
        x = rearrange(x, '(b n) c -> b n c', b=batch_size, n=num_nodes)
        x = torch.mean(x, dim=1)
        return x


class MHCAF(nn.Module):
    def __init__(self, dim_model, num_heads):
        super(MHCAF, self).__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads

        self.to_q = nn.ModuleList([nn.Linear(dim_model, dim_model) for _ in range(3)])
        self.to_k = nn.ModuleList([nn.Linear(dim_model, dim_model) for _ in range(3)])
        self.to_v = nn.ModuleList([nn.Linear(dim_model, dim_model) for _ in range(3)])

        self.recompute_q = nn.ModuleList([nn.Linear(2 * dim_model, dim_model) for _ in range(3)])
        self.recompute_k = nn.ModuleList([nn.Linear(2 * dim_model, dim_model) for _ in range(3)])
        self.recompute_v = nn.ModuleList([nn.Linear(2 * dim_model, dim_model) for _ in range(3)])

        self.multihead_attn_cross = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads)
        self.fc = nn.ModuleList([nn.Linear(2 * dim_model, dim_model) for _ in range(3)])

    def forward(self, tensors):
        g1, g2, g3 = tensors

        q1, k1, v1 = self.to_q[0](g1), self.to_k[0](g1), self.to_v[0](g1)
        q2, k2, v2 = self.to_q[1](g2), self.to_k[1](g2), self.to_v[1](g2)
        q3, k3, v3 = self.to_q[2](g3), self.to_k[2](g3), self.to_v[2](g3)

        cross_attn_output1_2, _ = self.multihead_attn_cross(q2.unsqueeze(0), k1.unsqueeze(0), v1.unsqueeze(0))
        cross_attn_output1_3, _ = self.multihead_attn_cross(q3.unsqueeze(0), k1.unsqueeze(0), v1.unsqueeze(0))
        combined_output1 = torch.cat([cross_attn_output1_2, cross_attn_output1_3], dim=2).squeeze(0)

        cross_attn_output2_1, _ = self.multihead_attn_cross(q1.unsqueeze(0), k2.unsqueeze(0), v2.unsqueeze(0))
        cross_attn_output2_3, _ = self.multihead_attn_cross(q3.unsqueeze(0), k2.unsqueeze(0), v2.unsqueeze(0))
        combined_output2 = torch.cat([cross_attn_output2_1, cross_attn_output2_3], dim=2).squeeze(0)

        cross_attn_output3_1, _ = self.multihead_attn_cross(q1.unsqueeze(0), k3.unsqueeze(0), v3.unsqueeze(0))
        cross_attn_output3_2, _ = self.multihead_attn_cross(q2.unsqueeze(0), k3.unsqueeze(0), v3.unsqueeze(0))
        combined_output3 = torch.cat([cross_attn_output3_1, cross_attn_output3_2], dim=2).squeeze(0)

        fusion_fea = torch.cat([
            self.fc[0](combined_output1),
            self.fc[1](combined_output2),
            self.fc[2](combined_output3)
        ], dim=1)

        return fusion_fea


class AtlasAttn(nn.Module):
    def __init__(self, num_channels):
        super(AtlasAttn, self).__init__()
        self.shared_fc = nn.Linear(num_channels, num_channels, bias=False)

    def forward(self, x):
        avg_pool = x.mean(dim=-1)
        max_pool, _ = x.max(dim=-1)

        avg_out = self.shared_fc(avg_pool)
        max_out = self.shared_fc(max_pool)

        combined_out = avg_out + max_out
        attention_weights = torch.sigmoid(combined_out).unsqueeze(-1)
        weighted_tensor = x * attention_weights
        return weighted_tensor


class NodeAttn(nn.Module):
    def __init__(self):
        super(NodeAttn, self).__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        attention_weights = self.sigmoid(self.conv(out))
        weighted_tensor = x * attention_weights
        return weighted_tensor


class AAWL(nn.Module):
    def __init__(self, num_channels):
        super(AAWL, self).__init__()
        self.atlasAttn = AtlasAttn(num_channels)
        self.nodeAttn = NodeAttn()

    def forward(self, x):
        x = torch.stack(x, dim=1)
        out = self.atlasAttn(x)
        out = self.nodeAttn(out)
        x = x + out
        x = x.view(x.size(0), -1)
        return x


class MAF_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_inputs):
        super(MAF_GNN, self).__init__()
        self.gcn_list = nn.ModuleList([GCN(input_dim[i], hidden_dim, hidden_dim) for i in range(num_inputs)])
        self.MHCAF_module = MHCAF(dim_model=hidden_dim, num_heads=2)
        self.AAWL_module = AAWL(num_channels=3)
        self.fc = nn.Linear(2 * hidden_dim * num_inputs, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, adjs):
        processed_graphs = []
        for gcn, x, adj in zip(self.gcn_list, inputs, adjs):
            processed_x = gcn(x, adj)
            processed_graphs.append(processed_x)

        x_a = self.AAWL_module(processed_graphs)
        x_m = self.MHCAF_module(processed_graphs)

        x = torch.cat((x_a, x_m), dim=1)
        x = self.fc(x)
        return x
