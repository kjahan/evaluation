import os

import pandas as pd

PATH = 'datasets'


def load(filename, delim='\t', date_col_name='datetime'):
	filename = os.path.join(PATH, filename)
	dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
	dataframe = pd.read_csv(filename, sep=delim, parse_dates=[date_col_name], date_parser=dateparse)
	return dataframe


def split(dataframe, split_time, date_col_name='datetime'):
	train_df = dataframe.loc[dataframe[date_col_name] <= split_time] 
	test_df = dataframe.loc[dataframe[date_col_name] > split_time]
	return train_df, test_df
