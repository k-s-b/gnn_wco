import datetime
import random
from collections import defaultdict
from typing import List, NamedTuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import Accuracy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from torch import Tensor
from torch.nn import BatchNorm1d, ModuleList
from torch_geometric.data import Data, DataLoader, Dataset, NeighborSampler
from torch_geometric.utils import from_networkx, to_undirected
from torch_sparse import SparseTensor
from tqdm import tqdm, tqdm_notebook, trange
from xgboost import XGBClassifier
from utils import process_leaf_idx
import dataset


class GraphData(object):
    def __init__(self,data, categories = ["HS6","importer.id"], use_xgb=True):
        self.data = data
        self.node_num = 0 
        self.edge_index = None
        self.edge_att = None # The edge type that connects two transaction
        self.categories = categories
        self.G = nx.Graph()
        
        # xgb config
        self.use_xgb = use_xgb
        self.num_trees = 100
        self.depth = 4
        
        # nodeid mapping
        self.n2id = dict()
        self.id2n = dict()
        if self.use_xgb:
            self.train_xgb_model()
        else:
            self.prepare_df()
        
    def train_xgb_model(self):
        """ Train XGB model """
        print("Training XGBoost model...")
        self.xgb = XGBClassifier(n_estimators=self.num_trees, max_depth=self.depth, n_jobs=-1, eval_metric="error")
        self.xgb.fit(self.data.dftrainx_lab, self.data.train_cls_label)   
    
        # Get leaf index from xgboost model 
        X_train_leaves = self.xgb.apply(self.data.dftrainx_lab)
        X_trainunlab_leaves = self.xgb.apply(self.data.dftrainx_unlab)
        X_valid_leaves = self.xgb.apply(self.data.dfvalidx_lab)
        X_test_leaves = self.xgb.apply(self.data.dftestx)
        
        # One-hot encoding for leaf index
        X_leaves = np.concatenate((X_train_leaves, X_trainunlab_leaves, X_valid_leaves, X_test_leaves), axis=0)
        transformed_leaves, self.leaf_dim, new_leaf_index = process_leaf_idx(X_leaves)
        train_rows = X_train_leaves.shape[0]
        trainunlab_rows = X_trainunlab_leaves.shape[0] + train_rows
        valid_rows = X_valid_leaves.shape[0] + trainunlab_rows
        self.train_leaves, self.trainunlab_leaves, self.valid_leaves, self.test_leaves = transformed_leaves[:train_rows],\
                                          transformed_leaves[train_rows:trainunlab_rows],\
                                          transformed_leaves[trainunlab_rows:valid_rows],\
                                          transformed_leaves[valid_rows:]
        
    def prepare_df(self):
        train_data = pd.concat((self.data.dftrainx_lab,self.data.dftrainx_unlab))
        self.leaf_dim = train_data.shape[1]
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_data)
        self.train_leaves = self.scaler.transform(self.data.dftrainx_lab)
        self.trainunlab_leaves = self.scaler.transform(self.data.dftrainx_unlab)
        self.valid_leaves = self.scaler.transform(self.data.dfvalidx_lab)
        self.test_leaves = self.scaler.transform(self.data.dftestx)
        
        
    def _getDF(self,stage):
        if stage == "train_lab":
            raw_df = self.data.train_lab
            feature = torch.LongTensor(self.train_leaves) if self.use_xgb else torch.FloatTensor(self.train_leaves)
            
        elif stage =="train_unlab":
            raw_df = self.data.train_unlab
            feature = torch.LongTensor(self.trainunlab_leaves) if self.use_xgb else torch.FloatTensor(self.trainunlab_leaves)
            
        elif stage == "valid":
            raw_df = self.data.valid_lab
            feature = torch.LongTensor(self.valid_leaves) if self.use_xgb else torch.FloatTensor(self.valid_leaves)
            
        elif stage == "test":
            raw_df = self.data.test
            feature = torch.LongTensor(self.test_leaves) if self.use_xgb else torch.FloatTensor(self.test_leaves)
        else:
            raise KeyError("No such stage for building dataframe")
        return raw_df, feature
    
    def _getNid(self,x):
        # get node index from raw data
        if x in self.id2n.keys():
            return self.id2n[x]
        else:
            self.id2n[x] = self.node_num
            self.n2id[self.node_num] = x
            self.node_num += 1
            return self.node_num - 1
        
    def _get_revenue(self,stage):
        if stage == "train_lab":
            return torch.FloatTensor(self.data.norm_revenue_train)
        elif stage =="train_unlab":
            return torch.ones(self.data.train_unlab.shape[0]).float()
        elif stage == "valid":
            return torch.FloatTensor(self.data.norm_revenue_valid)
        elif stage == "test":
            return torch.FloatTensor(self.data.norm_revenue_test)
        
    def get_AttNode(self,stage):
        nodes = [x for x,y in self.G.nodes(data=True) if y["att"]==stage]
        return nodes
        
    def _add_nodes(self,indices):
        for i in indices:
            self._getNid(i)
    
    def get_data(self,stage):
        pyg_data = Data()
        edges = []
        edge_att = []
        edge_label = []
        df, node_feature = self._getDF(stage)
        transaction_nodes = [self._getNid(i) for i in df.index]
        self.G.add_nodes_from(transaction_nodes, att = stage)
        target = torch.FloatTensor(df["illicit"].values)
        rev_target = self._get_revenue(stage)
        current_nodeNum = self.node_num
        for cid, cvalue in enumerate(self.categories):
            for gid, groups in df.groupby(cvalue):
                transaction_ids = list(groups.index)
                transaction_nodeid = [self._getNid(i) for i in transaction_ids]
                categoryNid = self._getNid(str(cvalue)+str(gid)) # convert to string incase duplication with transaction id
                
                # add node attribute
                self.G.add_node(categoryNid, att = cvalue)
                
                # create edges
                current_edges = list(zip(transaction_nodeid, [categoryNid] * len(transaction_nodeid)))
                self.G.add_edges_from(current_edges)
                edge_type = [cid] * len(transaction_nodeid)
                edge_target = groups["illicit"].values.tolist()
                edges.extend(current_edges)
                edge_att.extend(edge_type)
                edge_label.extend(edge_target)
                
        # append node feature (for categories)
        new_nodeNum = self.node_num - current_nodeNum
        init_feature = torch.zeros(new_nodeNum,self.num_trees) if self.use_xgb else torch.zeros(new_nodeNum,self.leaf_dim)
        init_feature = init_feature.long() if self.use_xgb else init_feature
        node_feature = torch.cat((node_feature,init_feature), dim=0)
        target = torch.cat((target, -torch.ones(new_nodeNum)))
        rev_target = torch.cat((rev_target, -torch.ones(new_nodeNum)))
                
        # PyG data format
        pyg_data.x = node_feature
        pyg_data.y = target
        pyg_data.rev = rev_target
        pyg_data.edge_index = torch.LongTensor(edges).T
        pyg_data.edge_index = torch.cat((pyg_data.edge_index, torch.flip(pyg_data.edge_index,[0])), dim=-1)
        pyg_data.edge_attr = torch.LongTensor(edge_att)
        pyg_data.edge_label = torch.FloatTensor(edge_label + edge_label)
        
        return pyg_data


def StackData(train_data, unlab_data, valid_data, test_data):
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


class Batch(NamedTuple):
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


class CustomData(LightningDataModule):
    def __init__(self,data,sizes, batch_size = 128):
        super(CustomData,self).__init__()
        self.data = data
        self.sizes = sizes
        self.valid_sizes = [-1 for i in self.sizes]
        self.batch_size = batch_size

    def train_dataloader(self):
        return NeighborSampler(self.data.train_edge, node_idx=self.data.train_idx,
                               sizes=self.sizes, return_e_id=True,
                               batch_size=self.batch_size,transform=self.convert_batch,
                               shuffle=True,
                               )

    def val_dataloader(self):
        return NeighborSampler(self.data.valid_edge, node_idx=self.data.valid_idx,
                               sizes=self.sizes, return_e_id=True,
                               batch_size=self.batch_size,transform=self.convert_batch,shuffle=False
                              )

    def test_dataloader(self):
        return NeighborSampler(self.data.test_edge, node_idx=self.data.test_idx,
                               sizes=self.sizes, return_e_id=True,
                               batch_size=self.batch_size,transform=self.convert_batch,shuffle=False
                              )

    def convert_batch(self, batch_size, n_id, adjs):
        return Batch(
            x=self.data.x[n_id],
            y=self.data.y[n_id[:batch_size]],
            rev = self.data.rev[n_id[:batch_size]],
            adjs_t=adjs,
        )
