import numpy as np
import pandas as pd

from event import SignalEvent

class Strategy:
    
    def __init__(self, strategyID) -> None:
        """
        Initialize strategy.

        Parameters:
        -= strategyID: unique identifier for strategy
        """
        self.strategyID = strategyID

    def calculate_signals(self, data) -> SignalEvent:
        symbol, action = self.BuyAndHold(data)

        if action != None:
            time = data[symbol]["timestamp"]

            signal = SignalEvent(   strategy_id = self.strategyID,
                                    symbol = symbol,
                                    datetime = time,
                                    signal_type = action,
                                    strength = 1,
                                    USD = 500,
                                    order_type = "MARKET"
                                )

            return signal
        
        return None

    def BuyAndHold(self, data):
        symbols = list(data.keys())

        symbol = symbols[0]
        # print(data)

        if len(data[symbol]) == 2500:
            return symbol, "BUY_LONG"

        if len(data[symbol]) == 5000:
            return symbol, "EXIT_LONG"
    
        if len(data[symbol]) == 6000:
            return symbol, "BUY_LONG"

        if len(data[symbol]) == 8000:
            return symbol, "EXIT_LONG"
            
        return None, None
