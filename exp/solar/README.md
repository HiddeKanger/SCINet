# Solar Active Regions

This folder contains experiments conducted on a dataset containing different features regarding solar active regions (SARs). The goal is to forecast the first column with the help of all columns in the dataset. As there is 16 of those, we, before training try to establish which are the most important and run experiments with a different number of columns. 


Files:

- `data/`: Here, the SAR training and test set should be placed, it can be downloaded here: https://drive.google.com/drive/u/2/folders/1Wgb2qyF-_aeq-WAW7KGKFOU_jfVLqIRS.
- `model_weights/`: contains all weights of the trained models.
- `utils/`: several supporting files used by the live trading system.
- `predictions`: Contains predictions made by different models on the test set.
- `train_live_model.ipynb`: Analyses the SAR data and trains several SCINet models on it.