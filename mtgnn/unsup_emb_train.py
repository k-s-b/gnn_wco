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
from torch import Tensor
from tqdm import tqdm, tqdm_notebook, trange
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from data import StackData, UnsupData
from models import SAGE 

class SageModel(pl.LightningModule):
    def __init__(self):
        super(SageModel, self).__init__()
        
        self.sage = SAGE( 
            in_channels=128, 
            hidden_channels=128, 
            num_layers=2, 
            leaf_len=leaf_len)
        
        self.X = None
        self.y = None
        
    def forward(self, x, adjs):
        
        return self.sage(x, adjs)
        
        
    def training_step(self, batch, batch_idx):
        
        x, y, rev, adjs = batch
        
        out = self.forward(x, adjs)
        
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
        out_y = y[:out.size(0)]

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        
#         print(out.shape, y.shape)
        if not self.X is None:
            self.X = torch.cat([self.X, out.detach().cpu()])
            self.y = torch.cat([self.y, out_y.flatten().detach().cpu()])
        else:
            self.X = out.detach().cpu()
            self.y = out_y.flatten().detach().cpu()
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y, rev, adjs = batch
        
        features = self.forward(x, adjs)
                
        return {'features':features, 'y': y.flatten()}
        
    def validation_epoch_end(self, outputs):
        
        valid_X = torch.cat([x["features"] for x in outputs], dim=0).detach().cpu()
        y_true = torch.cat([x["y"] for x in outputs], dim=0).detach().cpu()
        
        clf = LogisticRegression(max_iter=1e3)
        clf.fit(self.X, self.y)
        
        y_pred = clf.predict(valid_X)
        
        f1 = f1_score(y_true, y_pred, average='macro')
        
        self.log('f1', f1, prog_bar=True)
        
        self.X = None
        self.y = None
            
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None, on_tpu=False, using_native_amp=True, using_lbfgs=False):
                
        optimizer.step(closure=second_order_closure)

        self.scheduler.step(current_epoch + batch_nb / len(data_loader.train_dataloader()))
        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=0, last_epoch=-1, verbose=True)
                
        return optimizer


leaf_len = 1562
epochs=50
batch_size = 1024
sizes = [64, 32]

    
train_lab_data = torch.load('train_lab_data.pt')
train_unlab_data = torch.load('train_unlab_data.pt')
valid_data = torch.load('valid_data.pt')
test_data = torch.load('test_data.pt')

stacked_data = StackData(train_lab_data, train_unlab_data, valid_data, test_data)
data_loader = UnsupData(stacked_data, sizes = sizes, batch_size=batch_size)

model = SageModel()

early_stop_callback = EarlyStopping(
   monitor='f1',
   min_delta=0.00,
   patience=10,
   verbose=False,
   mode='max'
)

val_acc_callback = ModelCheckpoint(
        monitor='f1', 
        dirpath='unsup_checkpoints/',
        filename='{epoch:02d}-{f1:.4f}',
        save_last=True, 
        mode='max')

trainer = Trainer(
    gpus=[0],
#     accelerator='ddp',
#     progress_bar_refresh_rate=0,
    num_sanity_val_steps=0,
    max_epochs=epochs,
    callbacks=[val_acc_callback, early_stop_callback])

trainer.fit(model, data_loader)


