# GraphFC: Customs Fraud Detection with Label Scarcity

This repo contains the PyTorch implementation for "GraphFC: Customs Fraud Detection with Label Scarcity".  
The paper along with performance analysis on three real customs datasets can found <a href="https://arxiv.org/abs/2305.11377v1">here</a>



## Model Architecture of GraphFC
<img width="1025" alt="model architecture" src="https://user-images.githubusercontent.com/62580782/153579232-2ea4cac8-f17c-42ec-82bd-c68f304c0765.PNG">

Model architecture of GraphFC. Cross features extracted from GBDT step act as node features in the transaction graph. In the pre-training stage, GraphFC learns the model weights and refine the transaction representations. Afterwards, the model is fine-tuned with labeled data with dual-task learning framework to predict the illicitness and the additional revenue.


## How to train the model
The model code for GraphFC lies in `graph_sage` directory. 
Simply run `graph_sage/train.py` and specify the dataset parameters could train the model and evaluate the performance. 
Please refer to the scripts under the directory `run_*Data.sh`  for reproduce the results for individual country.
```
graph_sage
   |-- dataset.py -> Preprocess for customs data
   |-- models.py -> Main model modules
   |-- parser.py -> training arguments
   |-- pygData_util.py -> Data structure for graph data
   |-- run_Mdata.sh
   |-- run_Ndata.sh
   |-- run_Tdata.sh
   |-- train.py -> Train model
   |-- utils.py
```

## Arguments and Hyperparameters
```
# Dataset parameters
--data: Country name for building dataset ['synthetic', 'real-n', 'real-m', 'real-t']
--initial_inspection_rate: Initial inspection rate of labeled data
--train_from: Starting date of training data
--test_from: Starting date of testing data
--test_length: Number of days for testing data


# GraphFC Hyperparameters
--seed: Random seed
--epoch: number of epochs
--l2: l2 regularization 
--dim: dimension for hidden layers 
--lr: learning rate
--device: The device name for training, if train with cpu, please use:"cpu" 
```

## Data
You can experiment with GraphFC by downloading synthetic customs data from this [repo](https://github.com/Roytsai27/Dual-Attentive-Tree-aware-Embedding).
