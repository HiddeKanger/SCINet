from calendar import c
import datetime
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "utils/")

from event import MarketEvent

class HistoricCSVDataHandler:
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.
    """

    def __init__(self, data_dir, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form
        "symbol.csv", where symbol is a string in the list.

        Parameters:
        -= data_dir: Absolute directory path to the CSV files.
        -= symbol_list: A list of symbol strings.
        """
        self.data_dir = data_dir
        self.symbol_list = symbol_list
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.load_csv_files()

        self.update_bars()

    def load_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        """
        expected_column_names = [
                                    "timestamp",
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "volume",
                                ]
                                
        for symbol in self.symbol_list:
            #load CSV file with no header info
            self.symbol_data[symbol] = pd.read_csv(
                f"{self.data_dir}/{symbol}.csv",
                header = 0, 
                parse_dates = True,
                names = expected_column_names,
                usecols = range(len(expected_column_names)) #only load first few columns
            )

            print(f"Found dataset {symbol}.csv with size: {self.symbol_data[symbol].shape}")

        if len(self.symbol_list) > 1:

        #make sure datasets are of equal length and same timestamps        
            for symbol1 in self.symbol_list:
                for symbol2 in self.symbol_list:
                    if symbol1 == symbol2:
                        continue
                        
                    df = pd.merge(  self.symbol_data[symbol2], 
                                    self.symbol_data[symbol1],
                                    on = "timestamp")

                    self.symbol_data[symbol2] = df.iloc[:, :len(expected_column_names)]
                    self.symbol_data[symbol2].columns = expected_column_names
            
        # print(self.symbol_data)

    def _get_new_bar(self, symbol):
        """
        Returns generator to get latest bar from data feed
        """
        for _, b in self.symbol_data[symbol].iterrows():
            yield b

    def latest_bar(self, symbol):
        """
        Returns the last bar updated.
        """
        try:
            candles = self.latest_symbol_data[symbol]
        except KeyError:
            print("Symbol not found in database.")

        return candles[-1]

    def get_latest_bars(self, symbol, N=5):
        """
        Returns the last N bars updated.
        """
        try:
            candles = self.latest_symbol_data[symbol]
        except KeyError:
            print("Symbol not found in database.")
            
        return candles[-N:]

    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        try:
            candles = self.latest_symbol_data[symbol]
        except KeyError:
            print("Symbol not found in database.")

        return candles[-1]["timestamp"]

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        from the last bar.
        """
        try:
            candles = self.latest_symbol_data[symbol]
        except KeyError:
            print("Symbol not found in database.")

        return candles[-1][val_type]

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        try:
            candles = self.latest_symbol_data[symbol]
        except KeyError:
            print("Symbol not found in database.")
            
        return candles[-N:][val_type]

    def update_bars(self):
        """
        Pushes the latest bars to the latest_data for each symbol.
        """
        for symbol in self.symbol_list:
            try:
                bar = next(self._get_new_bar(symbol))
            except StopIteration:
                self.continue_backtest = False
            
            if bar is not None:
                if symbol in self.latest_symbol_data.keys():
                    self.latest_symbol_data[symbol].append(bar)
                else:
                    self.latest_symbol_data[symbol] = [bar]

        return self.latest_symbol_data

if __name__ == "__main__":
    DATA_DIR = "data/"
    SYMBOLS = ["BTCUSD"]

    #initialize different components
    data_handler = HistoricCSVDataHandler(  data_dir = DATA_DIR,
                                            symbol_list=SYMBOLS)
    i = 0

    print(len(data_handler.symbol_data[SYMBOLS[0]]))

    while True:
         
        # print(i)
        # if i % 100 == 0:
            # print(f"Iteration: {i}")

        #check if continue -> update the bars
        if data_handler.continue_backtest == True:
            latest_bar_data = data_handler.symbol_data[SYMBOLS[0]]
            # print(latest_bar_data)
        else:
            break

        i += 1