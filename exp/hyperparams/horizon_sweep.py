import pandas as pd
import numpy as np

import random
from time import time

from src.preprocess_data import preprocess
from train_scinet import train_scinet

from tqdm import tqdm

#expected dataformat of individual pairs
data_format = [
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                ]

#fraction of dataset used (could be 1, not that the first samples in the dataset are used)
fraction_used = 0.005

#train validation test set fractions of used data
train_frac = 0.7
val_frac = 0.2
test_frac = 0.1

OVERLAPPING = True
STANDARDIZE = True

RANDOM_SEED = None

if RANDOM_SEED != None:
    random.seed(RANDOM_SEED)

#names of pairs
pairs = ["BTCUSD"]

    # print(data[pair].isnull().values.any())

EXP = "horizon_sweep"
DIR = f"hyperparam_exp/{EXP}"

#X_LENS = [64, 80, 128, 160, 256]
#Y_LENS = [16, 16, 16, 16, 16]

hid_size = [1, 3, 5, 10]

for  hs in tqdm(hid_size):

    X_LEN = 256
    Y_LEN = 16

    data = {}

    for pair in pairs:
        data[pair] =  pd.read_csv(f"data/{pair}.csv")#.iloc[:10000, :] #debug

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
                            STANDARDIZE = STANDARDIZE
                            )

    for result in results:
        print(f"{result}: {results[result].shape}") 

    EPOCHS = 50
    BATCH_SIZE = 35

    N_BLOCKS = 2

    training_result  = train_scinet(   
                    X_train = results["X_train"],
                    y_train = results["y_train"],
                    X_val = results["X_val"],
                    y_val = results["y_val"],
                    X_test = results["X_test"],
                    y_test = results["y_test"],
                    epochs = EPOCHS,
                    batch_size = BATCH_SIZE,
                    X_LEN = X_LEN,
                    Y_LEN = [Y_LEN] * N_BLOCKS,
                    output_dim = [3, 1],
                    selected_columns = None,
                    hid_size = hs,
                    num_levels = 3,
                    kernel = [[3, 8, 13], [3]],
                    dropout = 0.5,
                    loss_weights= [0.4, 0.6],
                    probabilistic = False
                )

    model = training_result[0]
    history = training_result[1]
    print(history)
    print(history.history)
    history_pd = pd.DataFrame(history.history)


    model.save_weights(f"{DIR}/results/{'_'.join(pairs)}_{int(time())}_{hs}")
    history_pd.to_csv(f"{DIR}/results/{'_'.join(pairs)}_{int(time())}_{hs}.csv")
