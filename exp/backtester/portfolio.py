import numpy as np
import pandas as pd

from typing import Union

from event import FillEvent, MarketEvent, OrderEvent, SignalEvent
from performance import calc_sharpe_ratio, calc_drawdowns
from data_handler import HistoricCSVDataHandler

class Portfolio:
    """
    The Portfolio class handles the positions and market
    value of all instruments at a resolution of a "bar",
    i.e. secondly, minutely, hourly, etc. This is a
    base implentation without position-sizing or 
    risk-management.

    The positions DataFrame stores a time-index of the
    quantity of positions held.

    The holdings DataFrame stores the cash and total market
    holdings value of each symbol for a particular
    time-index, as well as the percentage change in
    portfolio total across bars.
    """

    def __init__(   self, 
                    bars: HistoricCSVDataHandler, 
                    initial_capital = 1000.0):
        """
        Initialises the portfolio with bars and an event queue.
        Also includes a starting datetime index and initial capital
        (USD unless otherwise stated).

        Parameters:
        -= bars: The DataHandler object with current market data.
        -= start_date: The start date (bar) of the portfolio.
        -= initial_capital: The starting capital in USD.
        """ 

        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.start_date = bars.get_latest_bar_datetime(bars.symbol_list[0])
        self.initial_capital = initial_capital

        self.historic_positions = self.construct_historic_positions()
        self.current_positions = dict( (k,v) for k, v in \
            [(s, 0) for s in self.symbol_list] )

        self.historic_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

    def construct_historic_positions(self) -> list:
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        d = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] )

        d["timestamp"] = self.start_date

        return [d]

    def construct_all_holdings(self) -> list:
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        """
        d = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] )

        d["timestamp"] = self.start_date
        d["cash"] = self.initial_capital
        d["fees"] = 0.0
        d["total"] = self.initial_capital

        return [d]

    def construct_current_holdings(self) -> dict:
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] )

        d["cash"] = self.initial_capital
        d["fees"] = 0.0
        d["total"] = self.initial_capital

        return d

    def update_time_index(self, latest_data: dict) -> None:
        """
        Adds a new record to the positions matrix for the current
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OHLCV).
        Makes use of a MarketEvent from the events queue.
        """
        random_symbol = list(latest_data.keys())[0]
        latest_datetime = latest_data[random_symbol]["timestamp"].iloc[-1]

        # Update positions
        # ================
        dp = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )
        dp["timestamp"] = latest_datetime
        
        for symbol in self.symbol_list:
            dp[symbol] = self.current_positions[symbol]

            # Append the current positions
            self.historic_positions.append(dp)

        # Update holdings
        # ===============
        dh = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )
        dh["timestamp"] = latest_datetime
        dh["cash"] = self.current_holdings["cash"]
        dh["fees"] = self.current_holdings["fees"]
        dh["total"] = self.current_holdings["cash"] #cash + position values

        for symbol in self.symbol_list:
            # Approximation to the real value
            market_value = self.current_positions[symbol] * latest_data[symbol]["close"].iloc[-1]
            dh[symbol] = market_value
            dh["total"] += market_value

        # Append the current holdings
        self.historic_holdings.append(dh)
    
    def update_positions_from_fill(self, fill: FillEvent) -> None:
        """
        Takes a FillEvent object and updates the position matrix to
        reflect the new position.

        Parameters:
        -= fill: FillEvent object to update the positions with.
        """

        # Update positions list with new quantities
        self.current_positions[fill.symbol] += \
            (fill.direction == "BUY")*fill.quantity - \
            (fill.direction == "SELL")*fill.quantity

        
    def update_holdings_from_fill(self, fill: FillEvent, latest_data: dict) -> None:
        """
        Takes a FillEvent object and updates the holdings matrix 
        to reflect the holdings value.

        Parameters:
        -= fill: FillEvent object to update the holdings with
        """
        # Update holdings list with new quantities
        exec_price = latest_data[fill.symbol]["close"].iloc[-1]
        cost = (fill.direction == "BUY") * exec_price * fill.quantity - \
                (fill.direction == "SELL") * exec_price * fill.quantity

        self.current_holdings[fill.symbol] += cost
        self.current_holdings["fees"] += fill.fees
        self.current_holdings["cash"] -= (cost + fill.fees)
        self.current_holdings["total"] -= (cost + fill.fees)

    def update_fill(self, fill: FillEvent, latest_data: dict) -> None:
        self.update_positions_from_fill(fill)
        self.update_holdings_from_fill(fill, latest_data)

    def request_order(self, signal: SignalEvent, latest_data: dict) -> Union[OrderEvent, None]:
        """
        Requests OrderEvent calculated from SignalEvent.

        Potentially taking risk management and 
        position-sizing into account.

        Parameters:
        -= signal - The tuple containing Signal information.
        """

        order = None

        symbol = signal.symbol
        direction = signal.signal_type
        USD = signal.USD

        if USD == "DEFAULT": 
            #open signal, portfolio manager must decide on size
            # if self.current_holdings["cash"] != 0:
            mkt_quantity = self.current_holdings["cash"]/latest_data[signal.symbol]["close"].iloc[-1]
            # else:
                # return None
            # print(mkt_quantity)
 
        else:
            #closed signal, USD amount fixed
            latest_close = latest_data[signal.symbol]["close"].iloc[-1]
            mkt_quantity = USD/latest_close

        curr_quantity = self.current_positions[symbol]

        if direction == "BUY_LONG":
            if mkt_quantity == 0.0:
                return None
            order = OrderEvent(symbol, signal.order_type, mkt_quantity, "BUY")
        if direction == "BUY_SHORT":
            order = OrderEvent(symbol, signal.order_type, mkt_quantity, "SELL")

        if direction == "EXIT_LONG":
            if curr_quantity == 0.0:
                return None
            order = OrderEvent(symbol, signal.order_type, abs(curr_quantity), "SELL")
        if direction == "EXIT_SHORT":
            order = OrderEvent(symbol, signal.order_type, abs(curr_quantity), "BUY")

        return order

    def create_equity_curve_dataframe(self) -> pd.DataFrame:
        """
        Creates a pandas DataFrame from the all_holdings
        list of dictionaries.
        """
        curve = pd.DataFrame(self.historic_holdings)
        # curve.set_index("timestamp", inplace=True)
        curve["returns"] = curve["total"].pct_change()
        curve["equity_curve"] = (1.0 + curve["returns"]).cumprod()

        return curve

    def output_summary_stats(self) -> Union[dict, pd.DataFrame]:
        """
        Creates a list of summary statistics for the portfolio.
        """
        equity_curve = self.create_equity_curve_dataframe()

        total_return = equity_curve["equity_curve"].iloc[-1]
        returns = equity_curve["returns"]
        pnl = equity_curve["equity_curve"]
        # print(total_return, returns, pnl)
        #hourly
        sharpe_ratio = calc_sharpe_ratio(\
                            returns, 
                            periods = 365*24*60
                        )

        drawdown, max_dd, dd_duration = calc_drawdowns(pnl)

        equity_curve["drawdown"] = drawdown
        
        stats = {}

        stats["Total Return"] = f"{(total_return - 1)*100:.2f}%"
        stats["Sharpe Ratio"] = f"{sharpe_ratio:.2f}"
        stats["Max Drawdown"] = f"{max_dd * 100.0:.2f}%"
        stats["Drawdown Duration"] = f"{dd_duration:.2f} bars"

        return stats, equity_curve