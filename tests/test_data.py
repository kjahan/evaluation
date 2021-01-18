from datetime import datetime
import pandas as pd

import src.dataset as dataset


def mock_data_with_date():
    f = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    dates = ['2020-08-01 10:24:01.621881', '2020-08-02 11:24:01.621881', '2020-08-03 17:24:01.621881',\
             '2020-08-04 15:24:01.621881', '2020-08-05 02:24:01.621881']
    df = pd.DataFrame.from_dict({'user_id': [1, 2, 3, 4, 5], 'item_id': [10, 20, 30, 40, 50], \
            'datetime': list(map(f, dates))})
    return df


def mock_data_with_timestamp():
    timestamps = ['1112486027', '1112484676', '1112484819', '1112484727']
    df = pd.DataFrame.from_dict({'user_id': [1, 2, 3, 4], 'item_id': [10, 20, 30, 40], \
            'timestamp': timestamps})
    return df


def test_split():
    df = mock_data_with_date()
    split_time = datetime.strptime('2020-08-04', '%Y-%m-%d')
    train, test = dataset.split(df, split_time)
    assert train.shape[0] == 3
    assert train.shape[0] + test.shape[0] == df.shape[0]


def test_split_empty_train():
    df = mock_data_with_date()
    split_time = datetime.strptime('2020-07-29', '%Y-%m-%d')
    train, test = dataset.split(df, split_time)
    assert train.empty
    assert test.shape[0] == df.shape[0]


def test_split_empty_test():
    df = mock_data_with_date()
    split_time = datetime.strptime('2020-08-06', '%Y-%m-%d')
    train, test = dataset.split(df, split_time)
    assert train.shape[0] == df.shape[0]
    assert test.empty


def test_parse_timestamp():
    df = mock_data_with_timestamp()
    parsed_df = dataset.parse_timestamp(df)
    assert df.shape[0] == parsed_df.shape[0]
    first_row_datetime_val = parsed_df.iloc[0]['datetime']
    assert first_row_datetime_val >= datetime.strptime('1970-01-01 00:00:00.000000', '%Y-%m-%d %H:%M:%S.%f')


def test_generate_true_labels():
    test_df = pd.DataFrame.from_dict({'user_id': [1, 3, 2, 2, 1], 'item_id': [10, 20, 30, 40, 50]})
    user_labels = dataset.generate_true_labels(test_df)
    assert len(user_labels[1]) == 2
    assert len(user_labels[2]) == 2
    assert len(user_labels[3]) == 1
    expected = [10, 50]
    assert all([a == b for a, b in zip(user_labels[1], expected)])
