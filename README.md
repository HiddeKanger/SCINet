# SCINet
TensorFlow SCINet implementation with extensions for Advances in Deep Learning Leiden University 2022

## Installation:
All code is Python 3.x native and depends on the packages listed in the `requirements.txt` file.

Setup environment:
`pip install requirements.txt`


## Usage
This repository contains a TensorFlow implementation of the SCINet model as described in https://arxiv.org/abs/2106.09305, adapted from https://github.com/cure-lab/SCINet. 

The repository is divided into two main folders: `base/` and `exp/`:
- `base/`: contains the core implementation of the SCINet model (and builder), the data preprocessing scripts and the training scripts. All experiments must use this base implementation.

- `exp/`: contains several experiments that contain Jupyter notebooks that guide the reader through several components of this work: training process and its preprocessing, the result reproduction of the original paper, the backtester, a live trading demo and more. All subfolders contain a more elaborate explanation of the experiments.

## SCINet Training
The SCINet training notebook serves as a first introduction to SCINet and introduces the different functions that define the workflow going from several large time series to making predictions on those series. To that end, the steps of data loading, preprocessing, training, predicting and evaluating are introduced. The dataset used in this process is a very easy self-generated dataset as to illustrate SCINet is able to learn.


## Results Reproduction
Here some of the results of the original SCINet paper are being reproduced. The original paper contained a lot of datasets with multiple hyperparameter combinations. 4 of them are being reproduced, 3 of which are of the different Electrical Transformer Temperature datasets (ETTh1, ETTh2, ETTm1). And the exchange rate dataset. For each dataset the best performing hyperparameter combination is chosen.

## Hyperparameter optimization


## Solar Dataset 
As an examplary application of SCINet it is applied on a dataset containing sixteen timeseries of different features of the sun with the objective of predicting the first feature. The set is believed to contain more signal as compared to the crypto datasets therefore serving as an intermediary between it and the generated dataset used in the training notebook. Particularly, the effect of column selection on the performance is investigated and results are compared to a naive benchmark.


## Backtester
The backtester is a system to assess SCINet's performance on a historic dataset in an event-driven manner. This is done by loading the historic data and replaying it as if it were live and repeatedly prompting the strategy for actions (buy, sell, do nothing, etc.).

A strategy is an implementation that takes the dataset as input and outputs actions. In this notebook, we will consider a simple strategy based on the output of the SCINet model. The exact strategy is explained in detail in the `backtester` subfolder.

The figure below shows the different individual components that make up the backtesting system. The data aggregation was done by simply downloading the crypto dataset: https://www.kaggle.com/datasets/jordialonsoesteve/cyrpto-data.
![backtester](exp/backtester/Backtester.png "Backtester system architecture")

## Live Trading Simulation
The live trading system works by subscribing to a brokers (FTX) websocket and processing all changes in the orderbook. This is filtered for the best bid and best ask in order to calculate the mid-price as ``the price''. No slippage or fees are taken into account as the small order assumption holds here (our market orders will have no impact on the price). This price fluctuations are accumulated for a certain period which then forms one data point for the SCINet model (open, high, low, close). These datapoints are accumulated to form a sequence that is the input to the SCINet model. Then, similar to the backtesting system. 

As we are dealing with a highly asynchronous system the different components described above are divided in three separate threads: websocket thread, data preparation thread, scinet model strategy thread. This way, the individual processes run in parallel allowing each of the processes to not be held back by the others.

The dlive demo shows three plots below each other: the best bid and ask (and halfway price) as retrieved from the websocket, the preprocessed data sample + prediction (in red below) and the live total equity from top to bottom respectively. 
![live_trading](exp/live_trading/dashboard.jpg "Live Trading System")



## Probabilistic SCINet
TODO