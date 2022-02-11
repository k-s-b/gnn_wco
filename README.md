# GraphFC: Customs Fraud Detection with Label Scarcity

This repo contains the PyTorch implementation for "GraphFC: Customs Fraud Detection with Label Scarcity"


## Model Architecture of GraphFC
<img width="1025" alt="model architecture" src="https://user-images.githubusercontent.com/62580782/153579232-2ea4cac8-f17c-42ec-82bd-c68f304c0765.PNG">

Model architecture of GraphFC. Cross features extracted from GBDT step act as node features in the transaction graph. In the pre-training stage, GraphFC learns the model weights and refine the transaction representations. Afterwards, the model is fine-tuned with labeled data with dual-task learning framework to predict the illicitness and the additional revenue.

## Data
You can experiment with GraphFC by downloading synthetic customs data from this [repo](https://github.com/Roytsai27/Dual-Attentive-Tree-aware-Embedding).
