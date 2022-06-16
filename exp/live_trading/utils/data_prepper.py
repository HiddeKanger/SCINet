import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from time import sleep, time

from collections import deque

class SCINET_data_prepper():
    def __init__(   self, 
                    ws_manager,
                    X_LEN,
                    Y_LEN,
                    update_period = 1):

        self.ws_manager = ws_manager
        self.markets = self.ws_manager.markets

        self.X_LEN = X_LEN
        self.Y_LEN = Y_LEN

        self.update_period = update_period #seconds

        self.running = False

        self.data = {}

        for market in self.markets:
            self.data[market] = {   "timestamp": deque([], maxlen = self.X_LEN + 1),
                                    "open": deque([], maxlen = self.X_LEN + 1),
                                    "high": deque([], maxlen = self.X_LEN + 1),
                                    "low": deque([], maxlen = self.X_LEN + 1),
                                    "close": deque([], maxlen = self.X_LEN + 1)
                                }

    def run(self):

        self.running = True
        self.start_timestamp = int(time())

        i = 1

        sleep(10)

        while self.running:
            begin_prices = {}

            for market in self.markets:
                bids = np.array(self.ws_manager.market_data[market]["bids"])
                asks = np.array(self.ws_manager.market_data[market]["asks"])

                if len(bids) == len(asks):
                    if not len(bids) > 0:
                        sleep(5)
                    
                    price = (bids + asks)/2
                    begin_prices[market] = price

            sleep(self.update_period)

            for market in self.markets:
                bids = np.array(self.ws_manager.market_data[market]["bids"])
                asks = np.array(self.ws_manager.market_data[market]["asks"])          

                if len(bids) == len(asks):
                    if not len(bids) > 0:
                        break
                    

                    start_price = begin_prices[market]
                    end_price = ((bids + asks)/2)

                    c = end_price[:len(start_price)] - start_price
                    idx_new_price_action = len(start_price) - len(c[c != 0])

                    price_action = end_price[idx_new_price_action:]

                    if len(price_action) == 0:
                        # print(start_price)
                        # print(end_price)
                        print(f"begin equals end with lengths: {len(start_price)} and {len(end_price)}")
                    else:
                        self.data[market]["timestamp"].append(self.start_timestamp + i)

                        self.data[market]["open"].append(price_action[0])
                        self.data[market]["high"].append(np.max(price_action))
                        self.data[market]["low"].append(np.min(price_action))
                        self.data[market]["close"].append(price_action[-1])

            # print(self.data)
            i += 1

    def stop(self):
        self.running = False