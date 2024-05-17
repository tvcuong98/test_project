# QM7-regression
Atomization energies prediction on QM7 dataset

#### What are the files?
`LG.py` is the linear regression solver for QM7.

`ds.py` implements the dataset class for QM7 dataset.

`models.py` contains the `torch` models for MLP and ElasticNet.

`MLP_ElasticNet.py` is the training script for MLP and ElasticNet.

`broken` folder contains the dataset, a model, and the training notebook for a GNN using `torch_geometric`. It achieves horrible results, because the defined model is probably not suitable for the task, so I did not include it in the report.
