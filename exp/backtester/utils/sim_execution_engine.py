from execution_engine import ExecutionEngine
from event import OrderEvent, FillEvent
import datetime

class SimulatedExecutionEngine(ExecutionEngine):
    """
    The simulated execution handler simulates OrderEvents
    being filled on a live market and produces FillEvents.
    This simulation can be easily extended to be as
    sophisticated as desired.
    """

    def __init__(self, fees):
        self.fees = fees
        super().__init__()

    def execute_order(self, event: OrderEvent) -> FillEvent:
        """
        Simulates order execution. Naive order execution,
        turns OrderEvent into FillEvent naively.

        Parameters: 
        -= fees: fee when executing order (in decimals so 1% -> 0.01)
        """

        assert event.type == "ORDER"

        fill_event = FillEvent(
                                timeindex = datetime.datetime.utcnow(),
                                symbol = event.symbol,
                                exchange = "DEFAULT",
                                quantity = event.quantity,
                                direction = event.direction,
                                fill_value = (1 - self.fees) * event.quantity,
                                fees = self.fees * event.quantity
                                )
        return fill_event