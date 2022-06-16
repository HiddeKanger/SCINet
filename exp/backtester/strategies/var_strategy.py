import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR
import statsmodels.api as sm

from event import SignalEvent

class VAR_Strategy:
    
    def __init__(self, strategyID, n_steps = 10) -> None:
        """
        Initialize strategy.

        Parameters:
        -= strategyID: unique identifier for strategy
        """
        self.strategyID = strategyID
        self.n_steps = n_steps

    def calculate_signals(self, data) -> SignalEvent:
        symbol, action = self.VAR_STRAT(data)

        if action != None:
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
        
        return None

    def VAR_STRAT(self, mvts):
        '''Function takes multivariate timeseries and predicts the next n_steps.
        Trend is first removed and then added back on at the end.'''
        symbol = list(mvts.keys())[0]

        if len(mvts[symbol]) >= 100:
            # print(mvts[symbol])
            mvts = mvts[symbol].iloc[:, 1:5]
            mvts = np.array(mvts)[-100:, :]
            # print(mvts.shape)
            # print(mvts)

            mvts_stationary = np.diff(mvts, axis = 0)

            model = VAR(mvts_stationary)

            results = model.fit(np.shape(mvts)[0]//2)
            best_order = results.k_ar

            forecast_diff = results.forecast(y=mvts_stationary[-best_order:,:],steps = self.n_steps).reshape((self.n_steps,-1))
            forecast = (mvts[-1,:] + np.cumsum(forecast_diff, axis = 0)).T
            
            # print(forecast[3, -1], forecast[:, -1], mvts[-1, 3])
            if forecast[3, -1] > mvts[-1, 3]:
                # print(f"long signal: {forecast[3, -1]} > {mvts[-1, 3]}")
                return symbol, "BUY_LONG"

            if forecast[3, -1] < mvts[-1, 3]:
                # print(f"sell signal: {forecast[3, -1]} < {mvts[-1, 3]}")

                return symbol, "EXIT_LONG"


            return forecast
        return None, None