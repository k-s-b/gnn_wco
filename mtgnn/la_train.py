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
from torch_scatter import scatter 
from torch import Tensor
from tqdm import tqdm, tqdm_notebook, trange
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import os
from models import LA_SAGE
from data import EmbedData
    
class LAModel(pl.LightningModule):
    def __init__(self):
        super(LAModel, self).__init__()
        
        self.la_sage = LA_SAGE(
            unsup_channels=128,
            in_channels=128, 
            hidden_channels=128, 
            num_layers=2, 
            leaf_len=leaf_len)
        self.la_sage.load_state_dict(torch.load('large_neighb_unsup_sage.pt'), strict=False)
        
    def forward(self, x, x_unsup, adjs):
        
        return self.la_sage(x, x_unsup, adjs)
            
    def training_step(self, batch, batch_idx):
        
        x, x_unsup, y_true, rev, adjs = batch
        
        y_pred = self.forward(x, x_unsup, adjs)

        loss = F.binary_cross_entropy_with_logits(y_pred.flatten(), y_true.flatten())
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, x_unsup, y_true, rev, adjs = batch
        
        y_pred = torch.sigmoid(self.forward(x, x_unsup, adjs))

        val_loss = F.binary_cross_entropy_with_logits(y_pred.flatten(), y_true.flatten())
        
        threshold = 0.5
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
                
        return {'y_pred':y_pred, 'y_true': y_true, 'val_loss': val_loss.item()}
        
    def validation_epoch_end(self, outputs):
        
        
        val_loss = np.mean([x["val_loss"] for x in outputs])
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0).detach().cpu()
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0).detach().cpu()
        
        
        f1 = f1_score(y_true, y_pred, average='macro')
       
        self.log('f1', f1, prog_bar=True)
        self.log('val_loss', val_loss, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        
        x, x_unsup, y_true, rev, adjs = batch
        
        logits = torch.sigmoid(self.forward(x, x_unsup, adjs))
        
#         y_true[torch.isnan(y_true)] = 0
#         threshold = 0.5
#         y_pred[y_pred >= threshold] = 1
#         y_pred[y_pred < threshold] = 0
#         print(logits)
        return {'logits': logits, 'y_true': y_true}
        
    def test_epoch_end(self, outputs):
        
        logits = torch.cat([x["logits"] for x in outputs], dim=0).detach().cpu().numpy().ravel()
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0).detach().cpu().numpy().ravel()
        
        
        f1 = f1_score(y_true, logits, average='macro')
        print('f1 ', f1)
        f, pr, re = self.torch_metrics(logits, y_true)
#         self.log('f1', f1, prog_bar=True)
#         self.log('val_loss', val_loss, prog_bar=True)
    
    def torch_metrics(self, y_prob,xgb_testy, best_thresh=None, display=True):
        """ Evaluate the performance"""
        pr, re, f = [], [], []
        # For validatation, we measure the performance on 5% (previously, 1%, 2%, 5%, and 10%)
        for i in [99,98,95,90]: 
            threshold = np.percentile(y_prob, i)
            precision = xgb_testy[y_prob > threshold].mean()
 
            recall = sum(xgb_testy[y_prob > threshold])/ sum(xgb_testy)

            f1 = 2 * (precision * recall) / (precision + recall)

            if display:
                print(f'Checking top {100-i}% suspicious transactions: {len(y_prob[y_prob > threshold])}')
                print('Precision: %.4f, Recall: %.4f' % (precision, recall))
            # save results
            pr.append(precision)
            re.append(recall)
            f.append(f1)
        return f, pr, re
            
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None, on_tpu=False, using_native_amp=True, using_lbfgs=False):
                
        optimizer.step(closure=second_order_closure)

        self.scheduler.step(current_epoch + batch_nb / len(data_loader.train_dataloader()))
        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=0, last_epoch=-1, verbose=True)
                
        return optimizer



leaf_len = 1562
epochs=25
                            
sizes = [-1, 200]
batch_size = 1024

stacked_data = torch.load('stacked_data.pt')
data_loader = EmbedData(stacked_data, sizes, batch_size)


model = LAModel()
# model = LAModel.load_from_checkpoint(checkpoint_path='la_checkpoints/epoch=07-f1=0.6329.ckpt')

early_stop_callback = EarlyStopping(
   monitor='f1',
   min_delta=0.00,
   patience=10,
   verbose=False,
   mode='max'
)

checkpoint_dir = "la_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
    
val_acc_callback = ModelCheckpoint(
        monitor='f1', 
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{f1:.4f}',
        save_last=True, 
        mode='max')

trainer = Trainer(
    gpus=[0],
#     progress_bar_refresh_rate=0,
    num_sanity_val_steps=0,
    max_epochs=epochs,
    callbacks=[val_acc_callback, early_stop_callback])

trainer.fit(model, data_loader)
trainer.test()
# trainer.test(model, test_dataloaders=data_loader.test_dataloader())

