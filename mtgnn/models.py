import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.data import Data, DataLoader, Dataset, NeighborSampler
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_scatter import scatter 
from torch import Tensor

class LAConv(SAGEConv):
    def __init__(self, label_predictor, in_channels, out_channels, **kwargs):
        super(LAConv, self).__init__(in_channels, out_channels, aggr='add', **kwargs)
        
        self.label_predictor = label_predictor
        
        # freeze label predictor
        for p in self.label_predictor.parameters():
            p.requires_grad = False
    
    def forward(self, x, edge_index):
        
        '''
        x - concat of x and unsupervised features
        '''

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x)
        
        # split to x, and unsupervised embeddings
        out_unsup = out[:,self.in_channels:]
        out = self.lin_l(out[:,:self.in_channels])

        x_r = x[1][:,:self.in_channels]

        out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return torch.cat([out, out_unsup], dim=-1)
    
    def message(self, x_j, x_i, index):

        '''
        i - central node that aggregates 
        x_j has shape [E, out_channels]
        index - indicies of the center nodes for each element in x_j
        '''

        # predicting similarity between center node and each of its neighbor 
        
        x_unsup_j = x_j[:,self.in_channels:]
        x_j = x_j[:,:self.in_channels]
        
        x_unsup_i = x_i[:,self.in_channels:]
        
        with torch.no_grad():
            A_j = self.label_predictor.edge_forward(x_unsup_i, x_unsup_j).detach()
                
        # sum all weights belonging to the same center node
        A_sum_scattered = scatter(A_j, index, dim=0, reduce="sum") 
        # unscattering sums
        A_sum = A_sum_scattered[index]
        # normalizing weights
        A_j = A_j / A_sum
        
        x_j = A_j * x_j
        x_j = torch.cat([x_j, x_unsup_j], dim=-1)
        
        return x_j
        
        
class LA_SAGE(nn.Module):
    def __init__(self, unsup_channels, in_channels, hidden_channels, num_layers, leaf_len):
        super(LA_SAGE, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        self.unsup_channels = unsup_channels
        
        self.label_predictor = LabelPredictor(unsup_channels, hidden_channels=128)
        self.label_predictor.load_state_dict(torch.load('la_predictor.pt'))
        
        self.emb = nn.Embedding(leaf_len, in_channels, padding_idx=0)
        self.ln = nn.LayerNorm(in_channels)
            
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(LAConv(self.label_predictor, in_channels, hidden_channels))
        
        self.linear = nn.Linear(hidden_channels,1)
        
    def forward(self, x, x_unsup, adjs):
        
        x = self.emb(x)

        x = torch.sum(x,dim=1) # summation over the leaves
        x = F.relu(self.ln(x))
        
        x = torch.cat([x, x_unsup], dim=-1) # cat learned features with fixed unsup embeddings

        for i, (edge_index, _, size) in enumerate(adjs):

            x_target = x[:size[1]]  # Target nodes are always placed first.

            x = self.convs[i]((x, x_target), edge_index)
            
            x_unsup = x[:,-self.unsup_channels:]
            x = x[:,:-self.unsup_channels]
            

            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
                
            x = torch.cat([x, x_unsup], dim=-1)
               
        x = x[:,:-self.unsup_channels]
        x = self.linear(x)
        
        return x



class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, leaf_len):
        super(SAGE, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        self.emb = nn.Embedding(leaf_len, in_channels, padding_idx=0)
        self.ln = nn.LayerNorm(in_channels)
        
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

        
    def forward(self, x, adjs):
        
        x = self.emb(x)

        x = torch.sum(x,dim=1) # summation over the leaves
        x = F.relu(self.ln(x))
        
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
                
        return x
    
class LabelPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LabelPredictor, self).__init__()
        
        self.lin1 = nn.Linear(in_channels * 3, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        
        self.ln = nn.LayerNorm(hidden_channels)

#         self.linear = nn.Linear(in_channels * 3, 1)
        
    def forward(self, x): 
      
        emb_a = x

        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        emb_b = x[index]
        
        emb_a = emb_a.detach()
        emb_b = emb_b.detach()
        
        emb_abs = torch.abs(emb_a - emb_b)
        emb_sum = emb_a + emb_b
        emb_mult = emb_a * emb_b
        
        x = torch.cat([emb_abs, emb_sum, emb_mult], dim=-1)

        x = F.relu(self.ln(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.lin2(x))


#         x = self.linear(x)
#         x = torch.sigmoid(x)
        
        return x, index
    
    def edge_forward(self, emb_a, emb_b):
                
        emb_abs = torch.abs(emb_a - emb_b)
        emb_sum = emb_a + emb_b
        emb_mult = emb_a * emb_b
        
        x = torch.cat([emb_abs, emb_sum, emb_mult], dim=-1)
        
        x = F.relu(self.ln(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.lin2(x))
        
        return x