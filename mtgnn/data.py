import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, NamedTuple, Optional

from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.data import Data, DataLoader, Dataset, NeighborSampler
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_scatter import scatter 
from torch import Tensor
from torch_cluster import random_walk
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.data import Data, DataLoader, Dataset, NeighborSampler
from pytorch_lightning import Trainer, LightningDataModule

def StackData(train_data, unlab_data, valid_data, test_data):
    '''
    stack pyG dataset.
    because the valid/test data should include train/unlab edges
    '''
    stack = Data()
    x, y, edge_index, edge_label, rev = [],[],[],[],[]
    
    # feature
    x.append(train_data.x)
    x.append(unlab_data.x)
    x.append(valid_data.x)
    x.append(test_data.x)
    x = torch.cat(x,dim=0)
    stack.x = x
    
    # target
    y.append(train_data.y)
    y.append(unlab_data.y)
    y.append(valid_data.y)
    y.append(test_data.y)
    y = torch.cat(y,dim=-1)
    stack.y = y
    
    # revenue
    rev.append(train_data.rev)
    rev.append(unlab_data.rev)
    rev.append(valid_data.rev)
    rev.append(test_data.rev)
    rev = torch.cat(rev,dim=-1)
    stack.rev = rev
    
    # edge index
    stack.train_edge = torch.cat((train_data.edge_index, unlab_data.edge_index), dim=1)
    stack.valid_edge = torch.cat((stack.train_edge,valid_data.edge_index ), dim=1)
    stack.test_edge = torch.cat((stack.valid_edge,test_data.edge_index ), dim=1)
    
    # transaction index
    stack.train_idx = train_data.node_idx
    stack.valid_idx = valid_data.node_idx
    stack.test_idx = test_data.node_idx
    
    return stack

def generate_unsup(data, sage):
    
    edge_index = torch.cat([data.train_edge, data.valid_edge, data.test_edge], dim=1)
    
    node_idx = torch.cat([data.train_idx, data.valid_idx, data.test_idx], dim=0)
    
    ns = NeighborSampler(edge_index, node_idx=node_idx,
                               sizes=[62,32], return_e_id=True,
                               batch_size=1024,
                               shuffle=False,
                               num_workers=16)
    
    data.x_unsup = torch.empty(data.x.size(0), 128)

    for batch_size, n_id, adjs in ns:

        x = data.x[n_id].to(device)
        adjs = [adj.to(device) for adj in adjs]

        with torch.no_grad():
            x_unsup = sage(x, adjs)
            
        data.x_unsup[n_id[:batch_size]] = x_unsup.detach().cpu()
    
    return data

class Batch(NamedTuple):
    '''
    convert batch data for pytorch-lightning
    '''
    x: Tensor
    y: Tensor
    rev: Tensor
    adjs_t: NamedTuple

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            rev=self.rev.to(*args, **kwargs),
            adjs_t=[(adj_t.to(*args, **kwargs), eid.to(*args, **kwargs), size) for adj_t, eid, size in self.adjs_t],
        )
    
class UnsupNeighborSampler(NeighborSampler):
    
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(UnsupNeighborSampler, self).sample(batch)
    
class UnsupData(LightningDataModule):
    def __init__(self,data,sizes, batch_size = 128):

        super(UnsupData,self).__init__()
        self.data = data
        self.sizes = sizes
        self.valid_sizes = [-1, 200]
        self.batch_size = batch_size
        
        
    def train_dataloader(self):
        return UnsupNeighborSampler(
                               self.data.train_edge, 
                               node_idx=self.data.train_idx,
                               sizes=self.sizes, 
                               return_e_id=True,
                               batch_size=self.batch_size,
                               shuffle=True,
                               drop_last=True,
                               transform=self.convert_batch,
                               num_workers=32)

    def val_dataloader(self):
        return NeighborSampler(
                               self.data.valid_edge, 
                               node_idx=self.data.valid_idx,
                               sizes=self.valid_sizes, 
                               return_e_id=True,
                               batch_size=self.batch_size,
                               shuffle=False,
                               drop_last=True,
                               transform=self.convert_batch,
                               num_workers=32)
    
    def test_dataloader(self):
        return NeighborSampler(
                               self.data.test_edge, 
                               node_idx=self.data.test_idx,
                               sizes=self.valid_sizes, 
                               return_e_id=True,
                               batch_size=self.batch_size,
                               shuffle=False,
                               drop_last=False,
                               transform=self.convert_batch,
                               num_workers=32)

    def convert_batch(self, batch_size, n_id, adjs):
        return Batch(
            x=self.data.x[n_id],
            y=self.data.y[n_id[:batch_size]],
            rev = self.data.rev[n_id[:batch_size]],
            adjs_t=adjs,
        )



# class UnsupSageNeighborSampler(NeighborSampler):
    
#     def __init__(self, x, y, sage, *args, **kwargs):
#         super(UnsupSageNeighborSampler, self).__init__(*args, **kwargs)
        
#         self.x = x
#         self.y = y
#         for p in self.sage.parameters():
#             p.requires_grad = False
    
#     def sample(self, batch):

#         batch_size, n_id, org_adjs = super(UnsupSageNeighborSampler, self).sample(batch)
        
#         _, unsup_n_id, adjs = super(UnsupSageNeighborSampler, self).sample(n_id)
#         with torch.no_grad():
#             unsup_emb = self.sage(self.x[unsup_n_id].to(device), adjs)
        
#         x = self.x[n_id]
#         x_unsup = unsup_emb.detach().cpu()
#         y = self.y[n_id[:batch_size]]
        
#         return x, x_unsup, y, org_adjs


class UnsupBatch(NamedTuple):
    '''
    convert batch data for pytorch-lightning
    '''
    x: Tensor
    x_unsup: Tensor
    y: Tensor
    rev: Tensor
    adjs_t: NamedTuple

    def to(self, *args, **kwargs):
        return UnsupBatch(
            x=self.x.to(*args, **kwargs),
            x_unsup=self.x_unsup.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            rev=self.rev.to(*args, **kwargs),
            adjs_t=[(adj_t.to(*args, **kwargs), eid.to(*args, **kwargs), size) for adj_t, eid, size in self.adjs_t],)

class EmbedData(LightningDataModule):
    def __init__(self, data, sizes, batch_size = 128):
        super(EmbedData,self).__init__()
        
        self.data = data
        self.sizes = sizes
        self.valid_sizes = sizes
        self.batch_size = batch_size
                
    def train_dataloader(self):
        return NeighborSampler(
                               self.data.train_edge, 
                               node_idx=self.data.train_idx,
                               sizes=self.sizes, 
                               return_e_id=True,
                               batch_size=self.batch_size,
                               shuffle=True,
                               drop_last=True,
                               transform=self.convert_batch,
                               num_workers=32)

    def val_dataloader(self):
        return NeighborSampler(
                               self.data.valid_edge, 
                               node_idx=self.data.valid_idx,
                               sizes=self.sizes, 
                               return_e_id=True,
                               batch_size=self.batch_size,
                               shuffle=False,
                               drop_last=True,
                               transform=self.convert_batch,
                               num_workers=32)
    def test_dataloader(self):
        return NeighborSampler(
                               self.data.test_edge, 
                               node_idx=self.data.test_idx,
                               sizes=self.sizes, 
                               return_e_id=True,
                               batch_size=self.batch_size,
                               shuffle=False,
                               drop_last=False,
                               transform=self.convert_batch,
                               num_workers=32)
    
    def convert_batch(self, batch_size, n_id, adjs):
        return UnsupBatch(
            x=self.data.x[n_id],
            x_unsup=self.data.x_unsup[n_id],
            y=self.data.y[n_id[:batch_size]],
            rev = self.data.rev[n_id[:batch_size]],
            adjs_t=adjs,
        )
    