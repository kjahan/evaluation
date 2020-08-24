from datetime import datetime
import pandas as pd

from src.data import split


def test_split():
    f = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    dates = ['2020-08-01 10:24:01.621881', '2020-08-02 11:24:01.621881', '2020-08-03 17:24:01.621881',\
             '2020-08-04 15:24:01.621881', '2020-08-05 02:24:01.621881']
    df = pd.DataFrame.from_dict({'user_id': [1, 2, 3, 4, 5], 'item_id': [10, 20, 30, 40, 50], \
            'datetime': list(map(f, dates))})
    split_time = datetime.strptime('2020-08-04', '%Y-%m-%d')
    train, test = split(df, split_time)
    assert train.shape[0] == 3
    assert train.shape[0] + test.shape[0] == df.shape[0]


def test_split_empty_train():
    f = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    dates = ['2020-08-01 10:24:01.621881', '2020-08-02 11:24:01.621881', '2020-08-03 17:24:01.621881',\
             '2020-08-04 15:24:01.621881', '2020-08-05 02:24:01.621881']
    df = pd.DataFrame.from_dict({'user_id': [1, 2, 3, 4, 5], 'item_id': [10, 20, 30, 40, 50], \
            'datetime': list(map(f, dates))})
    split_time = datetime.strptime('2020-07-29', '%Y-%m-%d')
    train, test = split(df, split_time)
    assert train.empty
    assert test.shape[0] == df.shape[0]


def test_split_empty_test():
    f = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    dates = ['2020-08-01 10:24:01.621881', '2020-08-02 11:24:01.621881', '2020-08-03 17:24:01.621881',\
             '2020-08-04 15:24:01.621881', '2020-08-05 02:24:01.621881']
    df = pd.DataFrame.from_dict({'user_id': [1, 2, 3, 4, 5], 'item_id': [10, 20, 30, 40, 50], \
            'datetime': list(map(f, dates))})
    split_time = datetime.strptime('2020-08-06', '%Y-%m-%d')
    train, test = split(df, split_time)
    assert train.shape[0] == df.shape[0]
    assert test.empty
