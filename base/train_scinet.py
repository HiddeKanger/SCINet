import numpy as np
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from time import time
import os
import sys

WORKDIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, WORKDIR_PATH)
from SCINet import scinet_builder


def check_Loss_Last(X_test, Y_test, data_type: str):
    pred_len = Y_test.shape[1]
    y_hat = np.repeat(X_test[:, -1, :], pred_len, 0).reshape(Y_test.shape)
    mae = tf.keras.losses.MeanAbsoluteError()
    print('....................')
    print('Loss using last X value in %s:' %data_type)
    print(mae(Y_test, y_hat))
    print('....................')

def train_scinet(   X_train: np.array,
                    y_train: np.array,
                    X_val: np.array,
                    y_val: np.array,
                    X_test: np.array,
                    y_test: np.array,
                    epochs = 100,
                    batch_size = 350,
                    X_LEN = 168,
                    Y_LEN = [24, 24, 24],
                    output_dim = [3, 3, 1],
                    selected_columns = [[3, 8, 13], [3, 8, 13], [3]],
                    hid_size= 32,
                    num_levels= 3,
                    kernel = 5,
                    dropout = 0.5,
                    loss_weights= [0.2, 0.2, 0.6],
                    learning_rate = 0.01,
                    probabilistic = False):


    print(f"===========================[SCINET]=====================================")
    print(f"Initializing training with data:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    #build SCINET model
    #callback: stops training when val_loss has been decreasing for past 25 epochs
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                                patience = 150, 
                                                restore_best_weights = True)


    input_dim = X_train.shape[2]
   # input_len, output_len = X_LEN, Y_LEN
    input_len = X_LEN
    model = scinet_builder(  output_len= Y_LEN,
                             output_dim= output_dim,
                             input_len = input_len,
                             input_dim= input_dim,
                             selected_columns= selected_columns,
                             hid_size= hid_size,
                             num_levels= num_levels,
                             kernel = kernel,
                             dropout = dropout,
                             loss_weights= loss_weights,
                             learning_rate = learning_rate, 
                             probabilistic = probabilistic)
    
    
    print(model.summary())

    # check_Loss_Last(X_train, y_train, 'training')
    # check_Loss_Last(X_val, y_val, 'validation')
    # check_Loss_Last(X_test, y_test, 'test')
    print(f"Is null X: {np.sum(np.isnan(X_train))}")
    print(f"Is null y: {np.sum(np.isnan(y_train))}")

    #fit model
    # SELECTED COLUMNS PASS TOWARDS THE LOSS AUTOMATICALLY HERE
    if selected_columns is not None:
        history = model.fit(
            X_train, 
            [y_train[:, :, selected_columns[i]] for i in range(len(selected_columns))], 
            epochs = epochs, 
            batch_size = batch_size, 
            validation_data = (X_val,
            [y_val[:, :, selected_columns[i]] for i in range(len(selected_columns))]), 
            callbacks=[callback]
            )
    else:
        history = model.fit(
            X_train, 
            [y_train] * len(output_dim), 
            epochs = epochs, 
            batch_size = batch_size, 
            validation_data = (X_val,
            [y_val] * len(output_dim)),
            callbacks=[callback]
            )
   

        #store performances

    return model, history, X_train , y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__": 

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
    fraction_used = 0.01

    #train validation test set fractions of used data
    train_frac = 0.7
    val_frac = 0.2
    test_frac = 0.1

    #predict next Y values based on previous X values
    X_LEN = 240
    Y_LEN = 24

    OVERLAPPING = True
    STANDARDIZE = True

    RANDOM_SEED = None

    if RANDOM_SEED != None:
        random.seed(RANDOM_SEED)

    #names of pairs
    pairs = ["BTCUSD"]
    data = {}

    for pair in pairs:
        data[pair] =  pd.read_csv(f"data/{pair}.csv")#.iloc[:10000, :] #debug
        # print(data[pair].isnull().values.any())

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

    EPOCHS = 1
    BATCH_SIZE = 350

    N_BLOCKS = 2
    
    training_result  = train_scinet(   
                    X_train = results["X_train"],
                    y_train = results["y_train"],
                    X_val = results["X_val"],
                    y_val = results["y_val"],
                    X_test = results["X_test"],
                    y_test = results["y_test"],
                    X_LEN = X_LEN,
                    Y_LEN = [Y_LEN] * N_BLOCKS,
                    epochs = EPOCHS,
                    batch_size = BATCH_SIZE,
                    output_dim = [results["X_train"].shape[2]] * N_BLOCKS,
                    selected_columns = None,
                    hid_size= 32,
                    num_levels= 3,
                    kernel = 5,
                    dropout = 0.5,
                    loss_weights= [0.4, 0.6],
                    probabilistic = False
                )

    model = training_result[0]

    model.save_weights(f"model_weights/{'_'.join(pairs)}_{int(time())}")