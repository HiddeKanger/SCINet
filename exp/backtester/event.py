class Event:
    """
    Event is base class providing an interface for all subsequent
    (inherited) events, that will trigger further events in the
    trading infrastructure.
    """
    pass


class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with
    corresponding bars.
    """

    def __init__(self):
        """
        Initializes the MarketEvent.
        """
        self.type = "MARKET"


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """

    def __init__(self, 
                strategy_id, 
                symbol, 
                datetime, 
                signal_type, 
                strength, 
                USD,
                order_type):
        """
        Initialises the SignalEvent.

        Parameters:
        -= strategy_id: The unique identifier for the strategy that
            generated the signal.
        -= symbol: The ticker symbol, e.g. "BTCUSD".
        -= datetime: The timestamp at which the signal was generated.
        -= signal_type: "BUY/SELL_LONG" or "BUY/SELL_SHORT".
        -= strength: An adjustment factor "suggestion" used to scale
            quantity at the portfolio level. Useful for pairs 
            strategies.
        -= USD: USD amount to buy of asset (closed signal) 
            or "DEFAULT" which indicates that the portfolio manager 
            must decide (open signal). 
        -= order_type: type of order to be executed (MARKET or LIMIT)
        """

        self.type = "SIGNAL"
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength
        self.USD = USD
        self.order_type = order_type

    def print_signal(self):
        """
        Outputs the values within the SignalEvent.
        """
        print(f"Strat_ID: {self.strategy_id}, Symbol: {self.symbol}, \
        Datetime: {self.datetime}, Signal_type: {self.signal_type}, \
            Strength: {self.strength}")


class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (e.g. GOOG), a type (market or limit),
    133
    quantity and a direction.
    """
    
    def __init__(self, symbol, order_type, quantity, direction):
        """
        Initialises the order type, setting whether it is
        a Market order ("MARKET") or Limit order ("LIMIT"), has
        a quantity (integral) and its direction ("BUY" or "SELL").
        
        Parameters:
        -= symbol: The instrument to trade.
        -= order_type: "MARKET" or "LIMIT".
        -= quantity: Non-negative integer for quantity.
        -= direction: "BUY" or "SELL" for long or short.
        """

        self.type = "ORDER"
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

    def print_order(self):
        """
        Outputs the values within the OrderEvent.
        """
        print(f"Symbol: {self.symbol}, \
        Order type: {self.order_type}, Quantity: {self.quantity}, \
            Direction: {self.direction}")


class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the fees of the trade from the brokerage.
    """

    def __init__(self, timeindex, symbol, exchange, quantity,
                direction, fill_value, fees):
        """
        Initialises the FillEvent object. Sets the symbol, exchange,
        quantity, direction, cost of fill and an optional
        fees.

        If fees is not provided, the Fill object will
        calculate it based on the trade size and exchange 
        fees.
        
        Parameters:
        -= timeindex: The bar-resolution when the order was filled.
        -= symbol: The instrument which was filled.
        -= exchange: The exchange where the order was filled.
        -= quantity: The filled quantity.
        -= direction: The direction of fill ("BUY" or "SELL")
        -= fill_value: The holdings value in dollars.
        -= fees: An optional fees parameter sent from exchange.
        """

        self.type = "FILL"
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_value
        self.fees = fees