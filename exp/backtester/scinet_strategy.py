import numpy as np
import pandas as pd

import tensorflow as tf
from event import SignalEvent

from preprocess_data import match_data

import sys

sys.path.insert(0, "base/")
from SCINet import scinet_builder

def construct_model_architecture(   input_dim,
                                    X_LEN = 240,
                                    Y_LEN = [24, 24, 24],
                                    output_dim = [15, 15, 1],
                                    hid_size = 32,
                                    num_levels = 3,
                                    kernel = 5,
                                    dropout = 0.5,
                                    loss_weights = [0.2, 0.2, 0.6],
                                    probabilistic = False):
                                    

    model = scinet_builder(  output_len = Y_LEN,
                             output_dim = output_dim,
                             input_len = X_LEN,
                             input_dim = input_dim,
                             hid_size = hid_size,
                             num_levels = num_levels,
                             kernel = kernel,
                             dropout = dropout,
                             loss_weights = loss_weights, 
                             probabilistic = probabilistic,
                             selected_columns = None)
    
    return model

def load_model( model_weights,
                n_features,
                X_LEN,
                Y_LEN):
    'Returns the model'

    model = construct_model_architecture(   input_dim = n_features,
                                            X_LEN = X_LEN,
                                            Y_LEN = Y_LEN)

    if model_weights != None:
        print(f"Loading in model weights... @ {model_weights}")
        model.load_weights(model_weights)
    else:
        print(f"WARNING: DID NOT SPECIFY MODEL WEIGHTS! BACKTESTING WITH RANDOM WEIGHT INIT.")

    return model

def preprocess_sample(data, X_LEN, data_format):

    matched_data = match_data(  data, 
                                list(data.keys()), 
                                data_format,
                                1.0)

    full_dataset = pd.DataFrame()

    for symbol in data:
        full_dataset = pd.concat([full_dataset, data[symbol]], axis = 1)

    sample = np.array(full_dataset[-X_LEN:])
 
    mean = np.dstack([np.mean(sample, axis = 0)] * X_LEN)[0, :, :].T
    std = np.dstack([np.std(sample, axis = 0)] * X_LEN)[0, :, :].T

    sample = (sample - mean)/std

    return np.array([sample])


class SCINET_Strategy:
    def __init__(self, strategyID, n_features, threshold = 0) -> None:
        """
        Initialize strategy.
        Parameters:
        -= strategyID: unique identifier for strategy
        """
        self.strategyID = strategyID

        self.X_LEN = 240
        self.Y_LEN = [24, 24, 24]

        self.input_len = self.X_LEN
        self.output_len = self.Y_LEN
        self.n_blocks = 2

        self.threshold = threshold

        self.model_weights = None

        self.n_features = n_features

        self.model = load_model(self.model_weights, self.n_features, self.X_LEN, self.Y_LEN)
        print(self.model.summary())

        self.data_format = [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                    ]

        self.planned_trades = {}

    def calculate_signals(self, data: dict) -> SignalEvent:
        symbol, action = self.strat(data)

        if action != None:
            # print(data[symbol])

            time = data[symbol]["timestamp"]
            signal = SignalEvent(   strategy_id = self.strategyID,
                                    symbol = symbol,
                                    datetime = time,
                                    signal_type = action,
                                    strength = 1,
                                    USD = "DEFAULT",
                                    order_type = "MARKET"
                                )

            return signal

    def strat(self, data: dict):
        symbol = list(data.keys())[0]
        timestamp = len(data[symbol]["timestamp"])

        if timestamp in self.planned_trades: #execute planned trades
            trade = self.planned_trades.pop(timestamp)

            symbol = trade["symbol"]
            type = trade["type"]

            return symbol, type
      
        if len(self.planned_trades) > 0: #if we have planned trades, do not make new ones
            return None, None

        if data[symbol].shape[0] < self.X_LEN: #only when enough data start doing stuff
            return None, None
        
        X = preprocess_sample(data.copy(), self.X_LEN, self.data_format)

        y_pred = self.model.predict(X)[2]
       
        #long_pred, short_pred = y_pred[0], y_pred[1]

        # close_prediction = short_pred[0, 0, 3]
        #print(y_pred)
        start_close_prediction = y_pred[0][0]
        end_close_prediction = y_pred[0][-1]

        # print(start_close_prediction, end_close_prediction)

        

        if end_close_prediction > start_close_prediction * (1 + self.threshold):
            self.planned_trades[timestamp + self.Y_LEN[0]] =  {
                                                    "type": "EXIT_LONG",
                                                    "symbol": symbol
                                                }

            return symbol, "BUY_LONG"

        if end_close_prediction < start_close_prediction * (1 - self.threshold):
            self.planned_trades[timestamp + self.Y_LEN[0]] =  {
                                                    "type": "EXIT_SHORT",
                                                    "symbol": symbol
                                                }

            return symbol, "BUY_SHORT"

        # if close_prediction > X[0, -1, 3]:
        #     print(f"Going long @ {data[symbol].shape[0]} Because {close_prediction} > {X[0, -1, 3]}")
        #     return symbol, "BUY_LONG"
        
        # if close_prediction < X[0, -1, 3]:
        #     print(f"Exiting long @ {data[symbol].shape[0]} Because {close_prediction} < {X[0, -1, 3]}")

        #     return symbol, "EXIT_LONG"

        return None, None