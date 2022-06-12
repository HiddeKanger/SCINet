from cmath import nan
from matplotlib.font_manager import json_dump
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os
import sys
import json

def match_data(data: dict, symbols: list, data_format: list, fraction: float) -> dict:
    '''This function makes sure the datasets are of equal length
    and match on the timestamp. If any dataset misses one datapoint
    this datapoint is removed for all datasets. ''' 

    print("Matching datasets and removing sparsity...")
    #match datasets removing sparsity
    for symbol1 in tqdm(symbols):
        for symbol2 in symbols:
            if symbol1 == symbol2:
                continue
                
            df = pd.merge(  data[symbol2], 
                            data[symbol1],
                            on = "timestamp")

            data[symbol2] = df.iloc[:, :len(data_format)]
            data[symbol2].columns = data_format

    #rename columns to include symbols (e.g. timestamp -> timestamp_BTCUSD)
    for symbol in symbols:
        new_data_format = []

        for column in data_format:
            new_data_format.append(f"{column}_{symbol}")

        data[symbol].columns = new_data_format

        #drop timestamp column, not interesting
        data[symbol] = data[symbol].drop(data[symbol].columns[0], 1)

        #only use up until fraction
        data[symbol] = data[symbol][:int(fraction * len(data[symbol]))]

    return data

def preprocess( data: dict, 
                symbols: list, 
                data_format: list, 
                fraction: float,
                train_frac: float,
                val_frac: float,
                test_frac: float,
                X_LEN: int,
                Y_LEN: int,
                OVERLAPPING: bool,
                STANDARDIZE: bool,
                standardization_settings: dict,
                ):
    
    print(f"Starting data preprocessing...")
    #data = match_data(data, symbols, data_format, fraction) #data sparsity removed

    full_dataset = pd.DataFrame()

    for symbol in data:
        full_dataset = pd.concat([full_dataset, data[symbol]], axis = 1)

    standardization_settings['total mean'] = np.mean(full_dataset, axis=0)
    standardization_settings['total std'] = np.std(full_dataset, axis=0)

    N_samples = len(full_dataset)

    print(f"Making train/validation/test splits...")
    #cut up dataset in different sets
    train_idx = int(train_frac * N_samples)
    val_idx = int(train_frac * N_samples) + int(val_frac * N_samples)
    test_idx = int(train_frac * N_samples) + int(val_frac * N_samples) + int(test_frac * N_samples)
    train_data = full_dataset[:train_idx]
    val_data = full_dataset[train_idx:val_idx]
    test_data = full_dataset[val_idx:test_idx]

    #create lists of tuples representing the samples
    train_samples = create_samples(train_data, X_LEN, Y_LEN, OVERLAPPING, STANDARDIZE)
    val_samples = create_samples(val_data, X_LEN, Y_LEN, OVERLAPPING, STANDARDIZE)
    test_samples = create_samples(test_data, X_LEN, Y_LEN, OVERLAPPING, STANDARDIZE)
    
    
    print(f"Making X-y splits...")
    #split in X and y for all three sets
    X_train = np.array([sample[0] for sample in train_samples])
    y_train = np.array([sample[1] for sample in train_samples])

    X_val = np.array([sample[0] for sample in val_samples])
    y_val = np.array([sample[1] for sample in val_samples])

    X_test = np.array([sample[0] for sample in test_samples])
    y_test = np.array([sample[1] for sample in test_samples])


    if STANDARDIZE:
        
        X_train, y_train, X_val, y_val, X_test, y_test = \
            standardize(standardization_settings, [[X_train,y_train],[X_val,y_val],[X_test,y_test]], X_LEN, Y_LEN)


    return {"X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,}

def create_samples(data: pd.DataFrame, X_LEN: int, Y_LEN: int, OVERLAPPING: bool, STANDARDIZE: bool) -> list:
    print(f"Making samples...")
    samples = []
    
    if OVERLAPPING:
        for i in tqdm(range(0, len(data) - X_LEN - Y_LEN, 1)): #loop until last sample, steps = 1
            samples.append([data[i: i + X_LEN], data[i + X_LEN: i + X_LEN + Y_LEN]])
    else:
        for i in tqdm(range(0, len(data) - X_LEN - Y_LEN, X_LEN + Y_LEN)): #step with X_LEN + Y_LEn
            samples.append([data[i: i + X_LEN], data[i + X_LEN: i + X_LEN + Y_LEN]])

    samples = np.array(samples)
    #standardize each individual sample
    #if STANDARDIZE:
    #    print(f"Standardizing samples...")
    #    for i, sample in tqdm(enumerate(samples)):
    #        #standardize X and y seperately!!!
    #        print(sample[0].shape)
    #        print(np.mean(sample[0], axis = 0))
    #        
    #        samples[i][0] = (sample[0] - np.mean(sample[0], axis = 0))/np.std(sample[0], axis = 0)
    #        samples[i][1] = (sample[1] - np.mean(sample[1], axis = 0))/np.std(sample[1], axis = 0)

    random.shuffle(samples)
    return samples

def standardize(st_settings, data, X_len, y_len):

    '''
    if st_settings['per_sample']:
    
        for dat in data:

            mean = np.mean(dat[0], axis = 1).reshape(dat[0].shape[0],1,dat[0].shape[2])
            std = np.std(dat[0], axis = 1).reshape(dat[0].shape[0],1,dat[0].shape[2])

            zero_std = np.where(std == 0)

            for da in dat:

                da = (da-mean) / std
                da = np.delete(da, zero_std[0], axis=0)
    '''
    if st_settings['per_sample']:

        for dat in data:

            mean = np.swapaxes(np.dstack([np.mean(dat[0], axis = 1)] * X_len), 1, 2)
            std = np.swapaxes(np.dstack([np.std(dat[0], axis = 1)] * X_len), 1, 2)
            std[std == 0] = 1

            dat[0] = (dat[0] - mean) / std
            dat[1] = (dat[1] - mean[:,:y_len,:]) / std[:,:y_len,:]

            std = np.std(dat[0], axis=1)
            results = np.where(std == 0)

            dat[0] = np.delete(dat[0], results[0], axis=0)
            dat[1] = np.delete(dat[1], results[0], axis=0)

    elif st_settings['leaky']:
        
        n_stocks = data[0][0].shape[2]

        for i in range(n_stocks):
            mean = st_settings['total mean'][i]
            std = st_settings['total std'][i]

            data[0][0][:,:,i] = (data[0][0][:,:,i] - mean) / std
            data[0][1][:,:,i] = (data[0][1][:,:,i] - mean) / std
            data[1][0][:,:,i] = (data[1][0][:,:,i] - mean) / std
            data[1][1][:,:,i] = (data[1][1][:,:,i] - mean) / std
            data[2][0][:,:,i] = (data[2][0][:,:,i] - mean) / std
            data[2][1][:,:,i] = (data[2][1][:,:,i] - mean) / std

    else:

        for dat in data:

            if st_settings['mode'] == 'log':

                dat[0] = np.sign(dat[0]) * np.log10(np.abs(dat[0])+1)
                dat[1] = np.sign(dat[1]) * np.log10(np.abs(dat[1])+1)

            elif st_settings['mode'] == 'sqrt':

                root_num = st_settings['sqrt_val']
                dat[0] = np.sign(dat[0]) * np.power(np.abs(dat[0]),1/root_num)
                dat[1] = np.sign(dat[1]) * np.power(np.abs(dat[1]),1/root_num)

            mean = np.mean(dat[0], axis = (0,1))
            std = np.std(dat[0] , axis = (0,1))
            dat[0] = (dat[0]-mean) / std
            dat[1] = (dat[1]-mean) / std

    return data[0][0], data[0][1], data[1][0], data[1][1], data[2][0], data[2][1]



    

if __name__ == "__main__":
    #expected dataformat of individual pairs
    '''
    data_format = [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                    ]
    '''
    data_format = [
                                "timestamp",
                                "price",
                    ]

    #fraction of dataset used (could be 1, not that the first samples in the dataset are used)
    fraction_used = 0.005

    #train validation test set fractions of used data
    train_frac = 0.7
    val_frac = 0.2
    test_frac = 0.1

    #predict next Y values based on previous X values
    X_LEN = 168
    Y_LEN = 20

    OVERLAPPING = True
    STANDARDIZE = True

    RANDOM_SEED = None

    if RANDOM_SEED != None:
        random.seed(RANDOM_SEED)

    #names of pairs
    pairs = ["ETTh1"]
    data = {}

    for pair in pairs:
        data[pair] =  pd.read_csv(os.path.realpath(__file__) + f"/data/Data_original/{pair}.csv").dropna()
        #data[pair] =  pd.read_csv(f"~/Documents/Studiedocumenten/2021-2022/ADL/Naamloos/Vincent/dataa/Data_original/{pair}.csv").dropna()

    # Process data:
    results = preprocess(   data = data, 
                            symbols = pairs,
                            data_format = data_format,
                            fraction = fraction_used,
                            train_frac = train_frac,
                            val_frac = val_frac,
                            test_frac = test_frac,
                            X_LEN = X_LEN,
                            Y_LEN = Y_LEN,
                            OVERLAPPING = OVERLAPPING,
                            STANDARDIZE = STANDARDIZE
                            )

    for result in results:
        print(f"{result}: {results[result].shape}") 

    print(results["X_train"])
    print(results["X_train"].shape)
    print(np.swapaxes(np.dstack([np.mean(results["X_train"], axis = 1)] * X_LEN), 1, 2).shape)

    file_name = "Data_preprocessed/ETTh1.csv"
    np.save(file_name, results)
