from matplotlib.font_manager import json_dump
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import math

WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../../"
sys.path.insert(1, WORKDIR_PATH)

from base.preprocess_data import preprocess
from base.train_scinet import train_scinet

WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, WORKDIR_PATH)

#============= Preprocessing ==============

#data_format = ["timestamp","open","high","low","close","volume",]
data_format = ["price"]
                    
fraction_used = 1
train_frac = 0.6
val_frac = 0.2
test_frac = 0.2

X_LEN = 48
Y_LEN = 24
RANDOM_SEED = 4321#None
OVERLAPPING = True
STANDARDIZE = True

standardization_settings = {'per_sample': True,
                            'leaky': False,
                            'mode': 'log', #only if per sample is false, choose from log, sqrt or lin
                            'sqrt_val': 2, #of course only if mode is sqrt
                            'total mean': [],
                            'total std': []}

pairs = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

#df = pd.read_csv(os.path.realpath(__file__) + f"/../data/Data_preprocessed/ETTh1.csv").dropna()
df = pd.read_csv(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2021-2022/ADL/SCINet_repo/exp/reprod/data/Data_preprocessed/ETTh1.csv").dropna()
df = df.swapaxes("index", "columns")

data = {}
for idx, pair in enumerate(pairs):
    data[pair] = df.iloc[idx]
 
results = preprocess(   data = data, 
                        symbols = pairs,
                        data_format = data_format,
                        fraction = fraction_used,
                        train_frac = train_frac,
                        val_frac = val_frac,
                        test_frac = test_frac,
                        X_LEN = X_LEN,
                        Y_LEN = Y_LEN,
                        OVERLAPPING = OVERLAPPING,
                        STANDARDIZE = STANDARDIZE,
                        standardization_settings = standardization_settings
                        )

#================ Training ====================

EPOCHS = 10
BATCH_SIZE = 8
HID_SIZE = 4
NUM_LEVELS = 3
KERNEL_SIZE = 5
DROPOUT = 0.5
PROBABILISTIC = False

LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001]

train_losses = np.zeros((len(LEARNING_RATES), EPOCHS))
val_losses = np.zeros((len(LEARNING_RATES), EPOCHS))
for idx, LEARNING_RATE in enumerate(LEARNING_RATES):

    model, history, X_train , y_train, X_val, y_val, X_test, y_test = train_scinet( X_train = results["X_train"].astype('float32'),
                                                                                    y_train = results["y_train"].astype('float32'),
                                                                                    X_val = results["X_val"].astype('float32'),
                                                                                    y_val = results["y_val"].astype('float32'),
                                                                                    X_test = results["X_test"].astype('float32'),
                                                                                    y_test = results["y_test"].astype('float32'),
                                                                                    epochs = EPOCHS,
                                                                                    batch_size = BATCH_SIZE,
                                                                                    X_LEN = X_LEN,
                                                                                    Y_LEN = [Y_LEN],
                                                                                    output_dim = [results["X_train"].shape[2]],
                                                                                    selected_columns = None,
                                                                                    hid_size= HID_SIZE,
                                                                                    num_levels= NUM_LEVELS,
                                                                                    kernel = KERNEL_SIZE,
                                                                                    dropout = DROPOUT,
                                                                                    loss_weights= [1],
                                                                                    learning_rate = LEARNING_RATE,
                                                                                    probabilistic = PROBABILISTIC)

    train_loss = history.history['loss']
    train_losses[idx] = train_loss

    val_loss = history.history['val_loss']
    val_losses[idx] = val_loss

plt.figure(figsize=(8,6))

X = np.arange(EPOCHS)
for idx, lr in enumerate(LEARNING_RATES):
    plt.plot(X, val_losses[idx], label=f'lr = {lr}')

plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Validation MAE', fontsize=15)
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.title('Learning rate comparison', fontsize=15)
plt.legend()
plt.savefig(f"Optimization_LearningRate.pdf")
plt.show()