import pandas as pd

def load_data(data_name, pairs):

    df = pd.read_csv('exp/reprod/data/Data_preprocessed/' + data_name + '.csv').dropna()
    df = df.swapaxes("index", "columns")

    data = {}

    for idx, pair in enumerate(pairs):
        data[pair] = df.iloc[idx]

    return data