from websocket import WebSocketApp
import time
import json

from collections import deque

class WebsocketManager:

    def __init__(self, URL, markets, max_len):
        self.URL = URL
        self.max_len = max_len

        self.ws = None

        assert len(markets) <= 4, "wow wow easy there..."

        self.markets = markets
        self.market_data = {}

        for market in self.markets:
            self.market_data[market] = {"times": deque([], maxlen = self.max_len),
                                        "bids": deque([], maxlen = self.max_len),
                                        "asks": deque([], maxlen = self.max_len),}


    def connect(self):
        self.ws = WebSocketApp( url = self.URL, 
                                on_open = self._wrap_callback(self.on_open),
                                on_message = self._wrap_callback(self.on_message),
                                on_close = self._wrap_callback(self.on_close),
                                on_error = self._wrap_callback(self.on_error),
                            )

        self.ws.run_forever()

    def _wrap_callback(self, f):
        def wrapped_f(ws, *args, **kwargs):
            if ws is self.ws:
                try:
                    f(ws, *args, **kwargs)
                except Exception as e:
                    raise Exception(f'Error running websocket callback: {e}')
                    
        return wrapped_f

    def on_open(self, ws):
        for market in self.markets:
            sub_msg = { "op": "subscribe", 
                                "channel": "ticker", 
                                "market": market, 
                            }

            ws.send(json.dumps(sub_msg))

    def on_message(self, ws, message):
        message = json.loads(message)

        if message["type"] == "update":

            market = message["market"]

            self.market_data[market]["times"].append(message["data"]["time"])
            self.market_data[market]["bids"].append(message["data"]["bid"])
            self.market_data[market]["asks"].append(message["data"]["ask"])

    def on_error(self, ws, error):
        print(error)
        
    def on_close(self, ws, close_status_code,close_msg):
        print("Websocket closed.")

    def stop(self):
        self.ws.close()

if __name__ == "__main__":
    URL = "wss://ftx.com/ws/"

    max_queue_len = 1000

    markets = ["BTC-PERP"]
    
    ws_manager = WebsocketManager(  URL = URL,
                                    markets = markets,
                                    max_len = max_queue_len)

    ws_manager.connect()

    