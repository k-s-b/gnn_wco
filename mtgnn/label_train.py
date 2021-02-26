import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, NamedTuple, Optional
import scipy.sparse as sp
# from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
import sys
sys.path.append('../graph_sage')
import parser
import dataset
import datetime
from datetime import timedelta
from custom_parser import get_parser
import numpy as np 
import pandas as pd 
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch_geometric
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.data import Data, DataLoader, Dataset, NeighborSampler
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_cluster import random_walk
from torch import Tensor
from tqdm import tqdm, tqdm_notebook, trange
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import os
from models import SAGE
from data import StackData


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
    

class SupData(LightningDataModule):
    def __init__(self,data,sizes, batch_size = 128):
        '''
        defining dataloader with NeighborSampler to extract k-hop subgraph.
        Args:
            data (Graphdata): graph data for the edges and node index
            sizes ([int]): The number of neighbors to sample for each node in each layer. 
                           If set to :obj:`sizes[l] = -1`, all neighbors are included
            batch_size (int): batch size for training
        '''
        super(SupData,self).__init__()
        self.data = data
        self.sizes = sizes
        self.valid_sizes = sizes #[i * 2 for i in self.sizes]
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

    def convert_batch(self, batch_size, n_id, adjs):
        return Batch(
            x=self.data.x[n_id],
            y=self.data.y[n_id[:batch_size]],
            rev = self.data.rev[n_id[:batch_size]],
            adjs_t=adjs,
        )

    
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
        
        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        
        return x
     

class LabelModel(pl.LightningModule):
    def __init__(self):
        super(LabelModel, self).__init__()
        
        self.unsup_sage = SAGE( 
            in_channels=128, 
            hidden_channels=128, 
            num_layers=2, 
            leaf_len=leaf_len)
        self.unsup_sage.load_state_dict(torch.load('large_neighb_unsup_sage.pt'))
        
        for p in self.unsup_sage.parameters():
            p.requires_grad = False
        
        
        self.label_predictor = LabelPredictor(in_channels=128, hidden_channels=128)
        
    def forward(self, x, adjs):
        
#         with torch.no_grad():
        x = self.unsup_sage(x, adjs)        
        return self.label_predictor(x)
                
    def training_step(self, batch, batch_idx):
        
        x, y, rev, adjs = batch
        
        y_pred, index = self.forward(x, adjs)

        y_true = (y == y[index]).float()

        loss = F.binary_cross_entropy(y_pred.flatten(), y_true.flatten())
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y, rev, adjs = batch
        
        threshold = 0.5
        
        y_pred, index = self.forward(x, adjs)

        y_true = (y == y[index]).float()

        val_loss = F.binary_cross_entropy(y_pred.flatten(), y_true.flatten())
        
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
                
        return {'y_pred':y_pred.flatten(), 'y_true': y_true.flatten(), 'val_loss': val_loss.item()}
        
    def validation_epoch_end(self, outputs):
        
        val_loss = np.mean([x["val_loss"] for x in outputs])
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0).detach().cpu()
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0).detach().cpu()
        
        
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        
        self.log('f1', f1, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_loss', val_loss, prog_bar=True)
        
            
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None, on_tpu=False, using_native_amp=True, using_lbfgs=False):
                
        optimizer.step(closure=second_order_closure)

        self.scheduler.step(current_epoch + batch_nb / len(data_loader.train_dataloader()))
        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=0, last_epoch=-1, verbose=True)
                
        return optimizer



leaf_len = 1562
epochs=25
batch_size = 1024
sizes = [-1, 200]


train_lab_data = torch.load('train_lab_data.pt')
train_unlab_data = torch.load('train_unlab_data.pt')
valid_data = torch.load('valid_data.pt')
test_data = torch.load('test_data.pt')

stacked_data = StackData(train_lab_data, train_unlab_data, valid_data, test_data)
data_loader = SupData(stacked_data, sizes = sizes, batch_size=batch_size)

model = LabelModel()

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=10,
   verbose=False,
   mode='min'
)

checkpoint_dir = "label_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
    
val_acc_callback = ModelCheckpoint(
        monitor='f1', 
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{f1:.4f}',
        save_last=True, 
        mode='max')

val_loss_callback = ModelCheckpoint(
        monitor='val_loss', 
#         dirpath=checkpoint_dir,
        filename='{epoch:02d}-{val_loss:.4f}',
        mode='min')

trainer = Trainer(
    gpus=[0],
#     progress_bar_refresh_rate=0,
    num_sanity_val_steps=0,
    max_epochs=epochs,
    callbacks=[val_acc_callback, val_loss_callback, early_stop_callback])

trainer.fit(model, data_loader)


