from datetime import datetime
import matplotlib.pyplot as plt

from utils.data_handler import HistoricCSVDataHandler
from utils.portfolio import Portfolio
from utils.sim_execution_engine import ExecutionEngine, SimulatedExecutionEngine 
from utils.strategy import Strategy

from strategies.scinet_strategy import SCINET_Strategy

from tqdm import tqdm
import numpy as np
import matplotlib.dates as md
import datetime as dt
import time as module_time

class Backtest:
    """
    Contains the settings and components for carrying out
    an event-driven backtest.
    """

    def __init__(   self,
                    data_handler: HistoricCSVDataHandler,
                    execution_engine: ExecutionEngine,
                    portfolio: Portfolio, 
                    strategy: Strategy,
                    START: int,
                    STOP: int,
                    STEP: int,
                    ) -> None:
        """
        Initialize backtest.
        Parameters:
        -= data_handler: HistoricCSVDataHandler class
        -= execution_engine: ExecutionEngine class
        -= portfolio: Portfolio class
        -= strategy: Strategy class
        """
        
        self.data_handler = data_handler
        self.execution_engine = execution_engine
        self.portfolio = portfolio
        self.strategy = strategy

        self.n_signals = 0
        self.n_orders = 0
        self.n_fills = 0

        self.START = START
        self.STOP = STOP
        self.STEP = STEP


    def run(self):
        """
        Execute the backtest.
        """

        print(f"Starting backtest @ {datetime.utcnow()} UTC")



        for i in tqdm(range(self.START, self.STOP, self.STEP)):
            latest_bar_data = self.data_handler.symbol_data.copy()
    
            for symbol in latest_bar_data:
                latest_bar_data[symbol] = latest_bar_data[symbol].iloc[self.START:i + 1, :]
            
        
            signal_event = self.strategy.calculate_signals(latest_bar_data)
            if signal_event != None:
                # print(signal_event)
                self.n_signals += 1

                order_event = self.portfolio.request_order(signal_event, latest_bar_data)
                
                if order_event != None:
                    # print(order_event)
                    self.n_orders += 1

                    fill_event = self.execution_engine.execute_order(order_event)
                    
                    if fill_event != None:
                        # print(fill_event)
                        self.n_fills += 1

                        self.portfolio.update_fill(fill_event, latest_bar_data)

            self.portfolio.update_time_index(latest_bar_data)

        self.print_performance(latest_bar_data)

    def print_performance(self, data: dict):
        """
        Outputs performance summary and stats
        """

        stats, equity_curve = self.portfolio.output_summary_stats()
        timestamps = np.array(equity_curve["timestamp"])[1:]/1000
        dates = [dt.datetime.fromtimestamp(ts) for ts in timestamps]
        time = md.date2num(dates)

        print("Stats summary:")
        print(stats)
        print() 

        print(f"Signal count: {self.n_signals}")
        print(f"Order count: {self.n_orders}")
        print(f"Fill count: {self.n_fills}")
        print()

        print(f"Creating equity curve chart...")

        plt.figure()
        ax=plt.gca()
        xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)

        # ax.plot(time, np.array(equity_curve["cash"])[1:], color = "lightgreen", label = "cash")
        # plt.plot(time, np.array(equity_curve["fees"])[1:], color = "orange", label = "fees")
        ax.plot(time, np.array(equity_curve["total"])[1:], color = "black", label = "total")
        # plt.plot(time, np.array(equity_curve["returns"])[1:], color ="lightblue", label = "returns")
        # plt.plot(time, equity_curve["equity_curve"], color = "purple", label = "equity_curve")
        # plt.plot(time, np.array(equity_curve["drawdown"])[1:], color = "red", label = "drawdown")

        ax.hlines(  self.portfolio.initial_capital, 
                    time[0],
                    time[-1],
                    color = "red",
                    linestyle = "--",
                    label = "initial equity")
        #print(data)
        pricedata = np.array(data["BTCUSD"]["close"])
        ax.plot(time, pricedata/pricedata[240] * self.portfolio.initial_capital, label = "B & H")
        
        ax.set_title("Equity curve")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Equity")
        plt.xticks( rotation=25 )

        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    DATA_DIR = "data/"
    SYMBOLS = ["BTCUSD", "ETHUSD", "LTCUSD"]
    fees = 0.0
    INIT_CAPITAL = 1000.0

    START = -int(1.5e4)
    STOP = -int(1)
    STEP = 1

    STRAT_ID = 0
    #initialize different components
    data_handler = HistoricCSVDataHandler(  data_dir = DATA_DIR,
                                        symbol_list=SYMBOLS
                                    )


    execution_engine = SimulatedExecutionEngine(fees)

    portfolio = Portfolio(  bars = data_handler,
                            initial_capital = INIT_CAPITAL)

    strategy = SCINET_Strategy( STRAT_ID, 
                                n_features = int(5 * len(SYMBOLS)),
                                threshold = 0.25)

    #run backtest
    backtester = Backtest(  data_handler = data_handler,
                            execution_engine = execution_engine,
                            portfolio = portfolio,
                            strategy = strategy,
                            START = START,
                            STOP = STOP, 
                            STEP = STEP)
    backtester.run()