from enum import unique
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def unique_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)

def match_data(data: dict, symbols: list, data_format: list, fraction: float) -> dict:
    '''This function makes sure the datasets are of equal length
    and match on the timestamp. If any dataset misses one datapoint
    this datapoint is removed for all datasets. ''' 

    if len(symbols) > 1:
        # print("Matching datasets and removing sparsity...")
        #match datasets removing sparsity
        for symbol1 in symbols:
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
    else:
        symbol = symbols[0]
        data[symbol] = data[symbol].drop(data[symbol].columns[0], 1)
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
                ):
    
    print(f"Starting data preprocessing...")
    data = match_data(data, symbols, data_format, fraction) #data sparsity removed

    full_dataset = pd.DataFrame()

    for symbol in data:
        full_dataset = pd.concat([full_dataset, data[symbol]], axis = 1)

    print(full_dataset.head(), full_dataset.shape)

    N_samples = len(full_dataset)

    print(f"Making train/validation/test splits...")
    #cut up dataset in different sets
    train_data = full_dataset[:int(train_frac * N_samples)]
    val_data = full_dataset[int(train_frac * N_samples):int((val_frac + train_frac) * N_samples)]
    test_data = full_dataset[int((val_frac + train_frac) * N_samples):]

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
        ## TRAINING SET
        mean = np.swapaxes(np.dstack([np.mean(X_train, axis = 1)] * X_LEN), 1, 2)
        std = np.swapaxes(np.dstack([np.std(X_train, axis = 1)] * X_LEN), 1, 2)
   
        X_train = (X_train - mean)/std
        y_train = (y_train - mean[:, :Y_LEN, :])/std[:, :Y_LEN, :]
 
        ## VALIDATION SET
        mean = np.swapaxes(np.dstack([np.mean(X_val, axis = 1)] * X_LEN), 1, 2)
        std = np.swapaxes(np.dstack([np.std(X_val, axis = 1)] * X_LEN), 1, 2)
    
        X_val = (X_val - mean)/std
        y_val = (y_val - mean[:, :Y_LEN, :])/std[:, :Y_LEN, :]

        ## TEST SET
        mean = np.swapaxes(np.dstack([np.mean(X_test, axis = 1)] * X_LEN), 1, 2)
        std = np.swapaxes(np.dstack([np.std(X_test, axis = 1)] * X_LEN), 1, 2)

        X_test = (X_test - mean)/std
        y_test = (y_test - mean[:, :Y_LEN, :])/std[:, :Y_LEN, :]


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
            X = data[i: i + X_LEN]
            y = data[i + X_LEN: i + X_LEN + Y_LEN]

            if True in unique_cols(X) or True in unique_cols(y):
                continue
            else:
                samples.append([X, y])
    else:
        for i in tqdm(range(0, len(data) - X_LEN - Y_LEN, X_LEN + Y_LEN)): #step with X_LEN + Y_LEN
            X = data[i: i + X_LEN]
            y = data[i + X_LEN: i + X_LEN + Y_LEN]

            if True in unique_cols(X) or True in unique_cols(y):
                continue
            else:
                samples.append([X, y])

    samples = np.array(samples)
    random.shuffle(samples)

    return samples

if __name__ == "__main__":
    #expected dataformat of individual pairs
    data_format = [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                    ]

    #fraction of dataset used (could be 1, not that the first samples in the dataset are used)
    fraction_used = 0.05

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
    pairs = ["BTCUSD", "ETHUSD", "LTCUSD"]
    data = {}

    for pair in pairs:
        data[pair] =  pd.read_csv(f"data/{pair}.csv").dropna()

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