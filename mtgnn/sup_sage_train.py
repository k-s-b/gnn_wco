import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, NamedTuple, Optional
from sklearn.metrics import f1_score, precision_score, recall_score
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch_geometric
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.data import Data, DataLoader, Dataset, NeighborSampler
from torch_geometric.nn import SAGEConv, MessagePassing
from torch import Tensor

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from data import StackData, UnsupData
from models import SAGE 

class SupSageModel(pl.LightningModule):
    def __init__(self):
        super(SupSageModel, self).__init__()
        
        self.sage = SAGE( 
            in_channels=128, 
            hidden_channels=128, 
            num_layers=2, 
            leaf_len=leaf_len)
        self.sage.load_state_dict(torch.load('large_neighb_unsup_sage.pt'))

        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Dropout(p=0.5),
            nn.Linear(64,1))
        
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
        
        
        y_pred = self.mlp(out)
        cls_loss = F.binary_cross_entropy_with_logits(y_pred.flatten(), out_y.flatten())
        
        loss = loss + cls_loss
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y, rev, adjs = batch
        
        emb = self.forward(x, adjs)
        y_pred = torch.sigmoid(self.mlp(emb))
                
        return {'y_pred': y_pred.flatten(), 'y_true': y.flatten()}
        
    def validation_epoch_end(self, outputs):
        
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0).detach().cpu()
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0).detach().cpu()
        
        threshold = 0.5
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        
        f1 = f1_score(y_true, y_pred, average='macro')
        
        self.log('f1', f1, prog_bar=True)
     
    def test_step(self, batch, batch_idx):
        
        x, y, rev, adjs = batch
        
        emb = self.forward(x, adjs)
        y_pred = torch.sigmoid(self.linear(emb))
                
        return {'y_pred': y_pred.flatten(), 'y_true': y.flatten()}
        
    def test_epoch_end(self, outputs):
        
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0).detach().cpu().numpy().ravel()
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0).detach().cpu().numpy().ravel()
        
        f, pr, re = self.torch_metrics(y_pred, y_true)
        
        threshold = 0.5
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        
        f1 = f1_score(y_true, y_pred, average='macro')
        print('f1 ', f1)
    
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
epochs=40
batch_size = 1024
sizes = [64, 32]

    
train_lab_data = torch.load('train_lab_data.pt')
train_unlab_data = torch.load('train_unlab_data.pt')
valid_data = torch.load('valid_data.pt')
test_data = torch.load('test_data.pt')

stacked_data = StackData(train_lab_data, train_unlab_data, valid_data, test_data)
data_loader = UnsupData(stacked_data, sizes = sizes, batch_size=batch_size)

model = SupSageModel()
# model = SupSageModel.load_from_checkpoint(checkpoint_path='sup_sage_checkpoints/epoch=14-f1=0.6426.ckpt')

early_stop_callback = EarlyStopping(
   monitor='f1',
   min_delta=0.00,
   patience=10,
   verbose=False,
   mode='max'
)

val_acc_callback = ModelCheckpoint(
        monitor='f1', 
        dirpath='sup_sage_checkpoints/',
        filename='{epoch:02d}-{f1:.4f}',
        save_last=True, 
        mode='max')

trainer = Trainer(
    gpus=[0],
    num_sanity_val_steps=0,
    max_epochs=epochs,
    callbacks=[val_acc_callback, early_stop_callback])

trainer.fit(model, data_loader)
trainer.test()
# trainer.test(model, test_dataloaders=data_loader.test_dataloader())


