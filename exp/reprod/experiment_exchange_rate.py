from pyexpat import XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE
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

from preprocess_data import preprocess
from base.train_scinet import train_scinet

WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, WORKDIR_PATH)


#data_format = ["timestamp","open","high","low","close","volume",]
data_format = ["price"]
                    
fraction_used = 0.002
train_frac = 0.6
val_frac = 0.2
test_frac = 0.2

X_LEN = 168
Y_LEN = 12

OVERLAPPING = True
STANDARDIZE = True

standardization_settings = {'per_sample': True,
                            'leaky': False,
                            'mode': 'log', #only if per sample is false, choose from log, sqrt or lin
                            'sqrt_val': 2, #of course only if mode is sqrt
                            'total mean': [],
                            'total std': []}


RANDOM_SEED = 4321#None

if RANDOM_SEED != None:
    random.seed(RANDOM_SEED)

pairs = ["exchange_rate_stock1", 
        "exchange_rate_stock2", 
        "exchange_rate_stock3",
        "exchange_rate_stock4",
        "exchange_rate_stock5",
        "exchange_rate_stock6",
        "exchange_rate_stock7",
        "exchange_rate_stock8"]

#df = pd.read_csv(os.path.realpath(__file__) + f"/../data/Data_preprocessed/exchange_rate.csv").dropna()
df=pd.read_csv(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2021-2022/ADL/SCINet_repo/exp/reprod/data/Data_preprocessed/exchange_rate.csv").dropna()
df = df.swapaxes("index", "columns")

data = {}
for idx, pair in enumerate(pairs):
    data[pair] = df.iloc[idx]

EPOCHS = 150
BATCH_SIZE = 4
HID_SIZE = 0.125
NUM_LEVELS = 3
KERNEL_SIZE = 5
DROPOUT = 0.5
LEARNING_RATE = 0.005
PROBABILISTIC = False

# Process data:
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
val_loss = history.history['val_loss']
target = 0.229 #value of MAE loss in paper

X = np.arange(len(train_loss))

plt.plot(X, train_loss, label='Training set')
plt.plot(X, val_loss, label="Validation set")
plt.axhline(y=target, color='r', linestyle='--', label="Paper's result")
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Mean absolute error', fontsize=15)
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.title('ETTm1', fontsize=15)
plt.legend()
plt.savefig(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2021-2022/ADL/SCINet_repo/exp/reprod/results/loss_exchange_rate_{Y_LEN}.pdf")
plt.show()

output = model(X_test)

stock = 0
total_timesteps = 1000

X_time_blocks = math.floor( total_timesteps / X_LEN )
Y_time_blocks = math.floor( total_timesteps / Y_LEN )

actual_prices = np.array([])
for t in range(X_time_blocks):
    actual_prices = np.append(actual_prices, X_test[t*X_LEN,:,stock])

predicted_prices = np.array([])
for t in range(Y_time_blocks):
    predicted_prices = np.append(predicted_prices, np.array(output[0][t*Y_LEN])[:,stock])

X_times = np.arange(len(actual_prices))
Y_times = np.arange(len(predicted_prices))
plt.plot(X_times, actual_prices, label='actual price')
plt.plot(Y_times, predicted_prices, label='predicted price')
plt.legend()
plt.savefig(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2021-2022/ADL/SCINet_repo/exp/reprod/results/predictions_exchange_rate_{Y_LEN}.pdf")
plt.show()