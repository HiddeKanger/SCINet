import numpy as np
import pandas as pd

def calc_sharpe_ratio(returns, periods = 365):
    """
    Create the Sharpe ratio for the strategy, based on a
    benchmark of zero (i.e. no risk-free rate information).

    Parameters:
    -= returns: A pandas Series representing period percentage returns.
    -= periods: Daily (365), Hourly (365*24), Minutely(365*24*60) etc.
    """
    
    return np.sqrt(periods) * np.mean(returns) / np.std(returns)

def calc_drawdowns(pnl):
    """
    Calculate the largest peak-to-trough drawdown of the PnL curve
    as well as the duration of the drawdown. Requires that pnl is 
    a pandas Series.
    
    Parameters:
    -= pnl - A pandas Series representing period percentage returns.
    
    Returns:
    -= drawdown: drawdown time series
    -= drawdown.max(): maximum peak-to-trough drawdown
    -= duration.max(): maximum duration of peak-to-trough
    """
    # Calculate the cumulative returns curve
    # and set up the High Water Mark (ATH of portfolio in time)
    hwm = [0]
    pnl = np.array(pnl)

    drawdown = np.zeros_like(pnl)
    duration =np.zeros_like(pnl)
    
    for t in range(1, len(pnl)):
        hwm.append(max(hwm[t-1], pnl[t])) 
        drawdown[t] = hwm[t] - pnl[t]
        duration[t] = (0 if drawdown[t] == 0 else duration[t - 1] + 1)

    return drawdown, drawdown.max(), duration.max()