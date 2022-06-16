from tracemalloc import start
import numpy as np
import pandas as pd

from time import sleep

import tensorflow as tf

from base.SCINet import SCINet

from collections import deque


def construct_model_architecture(   input_dim,
                                    input_len = 168,
                                    output_len = 20,
                                    n_blocks = 2):
    # Model cannot be loaded direclty, it has to be reconstructed and applied the weights:
    inputs = tf.keras.Input(shape = (input_len, input_dim))
        
    x_mid = SCINet( output_len= output_len,
                    input_len= input_len,
                    input_dim = input_dim,
                    num_levels = 2, 
                    single_step_output_One = False, name = 'Block1')(inputs)

    if n_blocks == 2:
        # New block
        new_input = tf.concat([inputs, x_mid], 1) # Concatenate
        new_input = tf.keras.layers.Cropping1D((output_len, 0))(new_input) # remove rest
        assert new_input.shape[1] == input_len # Make sure all is good

        x_final = SCINet(output_len= output_len,
        input_len= input_len,
        input_dim = input_dim,
        num_levels = 2, 
        single_step_output_One = True, name = 'Block2')(new_input)

        model = tf.keras.Model(inputs = inputs, outputs = [x_mid, x_final])
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
         loss = {'Block1' : "mae",
          'Block2' : "mae"})

    else:
        model = tf.keras.Model(inputs = inputs, outputs = x_mid)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), loss = "mae")
  
    return model

def load_model( model_weights,
                n_features,
                input_len = 168, 
                output_len = 20, 
                n_blocks = 2):
    'Returns the model'

    model = construct_model_architecture(   input_dim = n_features,
                                            input_len = input_len,
                                            output_len = output_len,
                                            n_blocks = n_blocks)

    if model_weights != None:
        print(f"Loading in model weights... @ {model_weights}")
        model.load_weights(model_weights)
    else:
        print(f"Input did not contain model weights, using random SCINET model")

    return model

def match_data(data: dict, symbols: list, data_format: list, fraction: float) -> dict:
    '''This function makes sure the datasets are of equal length
    and match on the timestamp. If any dataset misses one datapoint
    this datapoint is removed for all datasets. ''' 

    if len(symbols) > 1:
        # print("Matching datasets and removing sparsity...")
        #match datasets removing sparsity
        for symbol1 in symbols:
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    continue
                    
                df = pd.merge(  data[symbol2], 
                                data[symbol1],
                                on = "timestamp")

                data[symbol2] = df.iloc[:, :len(data_format)]
                data[symbol2].columns = data_format

        #rename columns to include symbols (e.g. timestamp -> timestamp_BTCUSD)
        for symbol in symbols:
            new_data_format = []

            for column in data_format:
                new_data_format.append(f"{column}_{symbol}")

            data[symbol].columns = new_data_format

            #drop timestamp column, not interesting
            data[symbol] = data[symbol].drop(data[symbol].columns[0], 1)

            #only use up until fraction
            data[symbol] = data[symbol][:int(fraction * len(data[symbol]))]
    else:
        data[symbols[0]] = data[symbols[0]].drop(data[symbols[0]].columns[0], 1)

    return data

def preprocess_sample(data, X_LEN, symbols, data_format):

    data = match_data(data, symbols, data_format, fraction=1.0)

    full_dataset = pd.DataFrame()

    for symbol in data:
        full_dataset = pd.concat([full_dataset, data[symbol]], axis = 1)

    sample = np.array(full_dataset[-X_LEN:])
 
    mean = np.dstack([np.mean(sample, axis = 0)] * X_LEN)[0, :, :].T
    std = np.dstack([np.std(sample, axis = 0)] * X_LEN)[0, :, :].T

    sample = (sample - mean)/std

    if any(np.isinf(sample).flatten()):
        return None

    return np.array([sample])

class LiveSCINET:
    def __init__(   self, 
                    prepper, 
                    n_features, 
                    X_LEN = 168, 
                    Y_LEN = 20, 
                    threshold = 0.05, 
                    model_weights = None,
                    begin_cash = 1000) -> None:
        """
        Initialize strategy.

        Parameters:
        -= strategyID: unique identifier for strategy
        """
        self.prepper = prepper
 
        self.n_features = n_features

        self.X_LEN = X_LEN
        self.Y_LEN = Y_LEN

        self.n_blocks = 2
        self.model_weights = model_weights
        self.threshold = threshold

        self.model = load_model(n_features = self.n_features,
                                model_weights = self.model_weights,
                                input_len = self.X_LEN,
                                output_len = self.Y_LEN,
                                n_blocks = self.n_blocks,
                                )

        print(self.model.summary())

        self.running = False

        self.data_format = [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                    ]

        self.current_times = deque([], maxlen = self.X_LEN + self.Y_LEN)
        self.current_sample = []

        self.prediction = None
        self.planned_trade = None

        self.cash = begin_cash
        self.position = None

        self.long_count = 0
        self.short_count = 0

        self.equity = deque([], maxlen = 100)

        self.historic_equity = []

    def run(self):
        self.running = True

        while self.running:

            times = self.prepper.data[self.prepper.markets[0]]["timestamp"]

            print(len(times), times.maxlen)

            if len(times) == times.maxlen:
                self.current_times.append(times[-1])

                data = {}

                for market in self.prepper.markets:

                    df = pd.DataFrame(self.prepper.data[market])
                    data[market] = df

                X = preprocess_sample(data, self.X_LEN, self.prepper.markets, self.data_format)
                self.current_sample = X[0, :, 3]
                print(f"CURRENT SAMPLE: {self.current_sample.shape}")
                if type(X) == type(None): #std = 0, do not enter new positions but close if needed
                    end_close_prediction = start_close_prediction = 1 

                else:
                    y_pred = self.model.predict(X)

                    long_pred, _ = y_pred[0], y_pred[1]

                    self.prediction = long_pred[0, :, 3]

                    start_close_prediction = long_pred[0, 0, 3]
                    end_close_prediction = long_pred[0, -1, 3]
               
                curr_price = data[self.prepper.markets[0]].iloc[-1, 3]
                print(curr_price)

                perc_pred = (end_close_prediction - start_close_prediction)/np.abs(start_close_prediction)

                if self.planned_trade == None and start_close_prediction != None and end_close_prediction != None:
                    #see if we must enter a position
                    if perc_pred > self.threshold:
                        self.position = {   "type": "LONG",
                                            "entry": curr_price,
                                            "amount": self.cash/curr_price}

                        self.planned_trade = {  "type": "EXIT_LONG",
                                                "exit": times[-1] + self.Y_LEN
                                            }

                        self.long_count += 1

                        print("Entered long.", self.position)
                        print(self.planned_trade)

                        self.cash = 0.0

                    if perc_pred < -self.threshold:
                        self.position = {   "type": "SHORT",
                                            "entry": curr_price,
                                            "amount": -self.cash/curr_price}

                        self.planned_trade = {  "type": "EXIT_SHORT",
                                                "exit": times[-1] + self.Y_LEN
                                            }

                        self.short_count += 1

                        print("Entered short.", self.position)
                        print(self.planned_trade)

                        self.cash = 0.0
                        
                    print(start_close_prediction, end_close_prediction, self.threshold)

                else:
                    print(f"Found planned exit, scheduled on: {self.planned_trade['exit']}. Now: {times[-1]}")
                    if times[-1] >=self.planned_trade["exit"]:
                        self.planned_trade = None
                        
                        position_worth = self.position_worth(curr_price)
                        self.cash += position_worth
                        self.position = None

                        print(f"exiting with position worth: {position_worth}")

                #update balances for graphs
                print(self.cash, self.position_worth(curr_price))
                self.equity.append(self.cash + self.position_worth(curr_price))
                self.historic_equity.append(self.equity[-1])

                print(self.equity[-1])

            sleep(self.prepper.update_period)

    def position_worth(self, curr_price):
        if self.position == None:
            return 0.0

        size = self.position["amount"]
        entry = self.position["entry"]
        exit = curr_price

        if size > 0:
            return size * curr_price
        if size < 0:
            return -size * (entry - exit) + -size * entry

    def stop(self):
        self.running = False