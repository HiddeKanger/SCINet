# Training SCINet

This folder contains an introduction to using SCINet. It discusses loading and preprocessing of the data and how to train SCINet and evaluate predictions afterwards. To that end, it uses a very simple and easy to predict datasets made up of some sine waves with added noise such that it is easy to see how good SCINet is performing.


Files:

- `data/`: contains the data used the train as well as the script used to generate the data.
- `model_weights/`: contains the weights of the trained model.
- `utils/`: supporting files that holds the plotting code.
- `images`: contains an explanatory image about the SCINet training pipeline.
- `train_scinet.ipynb`: explains loading, preprocessing, training and evaluating associated with SCINet 