from matplotlib.font_manager import json_dump
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import math
import keras
from preprocess_data import preprocess
from sklearn.metrics import mean_absolute_error as mae

WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../../"
sys.path.insert(1, WORKDIR_PATH)

from base.train_scinet import train_scinet
from exp.solar.utils.plotting import plot_per_timestep_mae

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

df = pd.read_csv(WORKDIR_PATH + "/data/Binance_BTCUSDT_minute.csv").dropna()
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

#================ Loading model =====================

model = keras.models.load_model(WORKDIR_PATH + '/saved_models/model_hidden_size_1')

#================ Predictions =======================

model_predictions = model(results['X_test'])
constant_predictions = np.stack((results['X_test'][:,-1, 0] for _ in range(Y_LEN)),axis = 1)

plot_per_timestep_mae(results['y_test'], model_predictions[:-1], constant_predictions, Y_LEN, ['Model'])