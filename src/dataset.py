import os
from datetime import datetime

import pandas as pd
import tqdm


def load(filename, path, delim='\t'):
    filename = os.path.join(path, filename)
    dataframe = pd.read_csv(filename, sep=delim)
    return dataframe


def parse_time(df):
    date_parse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    df['datetime'] = df['datetime'].apply(date_parse)
    return df


def parse_timestamp(df):
    ts_parse = lambda x: datetime.utcfromtimestamp(int(x))
    df['datetime'] = df['timestamp'].apply(ts_parse)
    df = df.drop('timestamp', 1)
    return df


def split(dataframe, split_time):
    """
    dataframe: pandas dataframe with 
    """
    train_df = dataframe.loc[dataframe['datetime'] <= split_time]
    test_df = dataframe.loc[dataframe['datetime'] > split_time]
    return train_df, test_df


def generate_true_labels(test_df):
    user_labels = {}
    with tqdm.tqdm(total=test_df.shape[0]) as progress:
        for index, row in test_df.iterrows():
            user_id, item_id = row['user_id'], row['item_id']
            try:
                item_id = int(item_id)
                user_labels[user_id].append(item_id)
            except KeyError:
                user_labels[user_id] = [item_id]
            progress.update(1)
        print("No of users in test fold: {}".format(len(user_labels.keys())))
    return user_labels
