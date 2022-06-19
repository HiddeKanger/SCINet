# Live Simulated Trading

This folder contains the files used for a live trading example, as shown in the presentation. The live trading system works by subscribing to a brokers (FTX) websocket and processing all changes in the orderbook. This is filtered for the best bid and best ask in order to calculate the mid-price as ``the price''. No slippage or fees are taken into account as the small order assumption holds here (our market orders will have no impact on the price). This price fluctuations are accumulated for a certain period which then forms one data point for the SCINet model (open, high, low, close). These datapoints are accumulated to form a sequence that is the input to the SCINet model. Then, similar to the backtesting system, the performance is tracked and plotted live.


Files:

- `data/`: contains all data used by the live trading model (used for training the live model)
- `model_weights/`: contains all weights of the trained models
- `utils/`: several supporting files used by the live trading system
- `live_data.py`: python file stand-alone version of the notebook, can by run separately
- `LiveDemo.ipynb`: notebook for running a trained live model on live data
- `train_live_model.ipynb`: train a model on data in the `data/` folder that can then be used for live trading