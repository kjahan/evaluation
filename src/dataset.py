import os
from datetime import datetime

import pandas as pd

PATH = 'datasets'


def load(filename, path=PATH, delim='\t'):
    filename = os.path.join(path, filename)
    dataframe = pd.read_csv(filename, sep=delim)
    return dataframe


def parse_time(df):
    date_parse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    df['datetime'] = df['datetime'].apply(date_parse)
    return df


def parse_ts(df):
    ts_parse = lambda x: datetime.utcfromtimestamp(int(x))
    df['datetime'] = df['timestamp'].apply(ts_parse)
    return df


def split(dataframe, split_time):
    train_df = dataframe.loc[dataframe['datetime'] <= split_time]
    test_df = dataframe.loc[dataframe['datetime'] > split_time]
    return train_df, test_df
