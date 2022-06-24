# Hyperparameter Optimization


This folder contains the files used for hyperparameter optimization using the SCINet model on cryptocurrency data, in particular Bitcoin.
Only the hyperparameters that are directly accessible via the 'train_scinet' or 'preprocess' functions are being investigated, that is, the learning rate and the hidden size.
The models of all the hyperparameter settings are saved so that they are accessible for later purposes. After the training part each model's performance is evaluated.


Files:

- `data/`: contains the cryptocurrency data of Bitcoin used to train and evaluate the models
- `saved_models/`: contains all the saved trained models for the different hyperparameters
- `utils/`: several supporting files used by the hyperparameter tuning
- `results/`: results, plots outputted by the scripts in this folder
- `evaluation.py`: python file stand-alone version of the evaluation part of the notebook, can be run separately
- `hidden_size.py`: python file stand-alone version of the training and comparing of several settings for the hidden_size parameter
- `learning_rate_.py`: python file stand-alone version of the training and comparing of several settings for the learning_rate parameter
- `hyperparameter_tuning.ipynb`: train a model on data in the `data/` folder that can then be used for hyperparameter tuning toggling between different variational parameters for the hyperparameters, and evaluating and comparing their results.