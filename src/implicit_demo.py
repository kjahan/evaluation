import argparse
import codecs
import logging
import time

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import tqdm

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (
    AnnoyAlternatingLeastSquares,
    FaissAlternatingLeastSquares,
    NMSLibAlternatingLeastSquares,
)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.datasets.lastfm import get_lastfm
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (
    BM25Recommender,
    CosineRecommender,
    TFIDFRecommender,
    bm25_weight,
)


import pandas as pd
import os
from datetime import datetime


import sys
# sys.path.append("/Users/jahan/workspace/evaluation/src/")
import dataset
import metrics


# maps command line model argument to class name
MODELS = {
    "als": AlternatingLeastSquares,
    "nmslib_als": NMSLibAlternatingLeastSquares,
    "annoy_als": AnnoyAlternatingLeastSquares,
    "faiss_als": FaissAlternatingLeastSquares,
    "tfidf": TFIDFRecommender,
    "cosine": CosineRecommender,
    "bpr": BayesianPersonalizedRanking,
    "lmf": LogisticMatrixFactorization,
    "bm25": BM25Recommender,
}


def get_model(model_name):
    print("getting model %s" % model_name)
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % model_name)

    # some default params
    if model_name.endswith("als"):
        params = {"factors": 64, "dtype": np.float32}
    elif model_name == "bm25":
        params = {"K1": 100, "B": 0.5}
    elif model_name == "bpr":
        params = {"factors": 63}
    elif model_name == "lmf":
        params = {"factors": 30, "iterations": 40, "regularization": 1.5}
    else:
        params = {}

    return model_class(**params)


def load_data(fn, data_path, spit_date):
    ratings_ = dataset.load(fn, path=data_path, delim=',')
    ratings = dataset.parse_timestamp(ratings_)
    # rename ratings columns
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id", "rating": "rating",  "datetime": "datetime"})
    # Movielese data stats
    print("ratings columns: {}".format(ratings.columns))
    print("No of rows in ratings df: {}".format(ratings.shape[0]))
    print("Min datetime: {}, max datetime: {}".format(ratings["datetime"].min(), 
                                                      ratings["datetime"].max()))
    split_time = pd.datetime.strptime(spit_date, '%Y-%m-%d %H:%M:%S.%f')
    # split train/test folds
    train_df, test_df = dataset.split(ratings, split_time)
    print("Size of train dataset: {} & size of test dataset: {}".format(train_df.shape[0], test_df.shape[0]))
    print(ratings.head(5))
    return train_df, test_df


def train_als(train_df, test_df, min_rating = 4.0):
    # map each user/item to a unique numeric value
    train_df['user_id'] = train_df['user_id'].astype("category")
    train_df['item_id'] = train_df['item_id'].astype("category")

    ratings_csr = coo_matrix((train_df['rating'].astype(np.float32), 
                             (train_df['item_id'].cat.codes.copy(), 
                              train_df['user_id'].cat.codes.copy()))).tocsr()

    items = np.array(train_df['item_id'].cat.categories)
    users = np.array(train_df['user_id'].cat.categories)
    ratings = ratings_csr
    
    # remove things < min_rating, and convert to implicit dataset
    # by considering ratings as a binary preference only
    ratings.data[ratings.data < min_rating] = 0
    ratings.eliminate_zeros()
    ratings.data = np.ones(len(ratings.data))
    model = AlternatingLeastSquares()
    # lets weight these models by bm25weight.
    ratings = (bm25_weight(ratings, B=0.9) * 5).tocsr()
    # train the model
    start = time.time()
    model.fit(ratings)
    print("Training time: {}".format(time.time() - start))
    return model, users, items, ratings


def generate_user_recommendations(model, user_labels, train_username_to_id_map, training_ds):
    items, users, ratings = training_ds
    user_ratings = ratings.T.tocsr()
    user_recommendations = {}
    N = 10
    with tqdm.tqdm(total=len(user_labels)) as progress:
        for username in user_labels.keys():
            try:
                user_id = train_username_to_id_map[username]
                recomms_ = model.recommend(user_id, user_ratings, N)
                recomms = [(int(items[item_id]), score) for item_id, score in recomms_]
                user_recommendations[username] = recomms
            except:
                continue
            progress.update(1)
    return user_recommendations


def run():
    fn = "ratings.csv"
    data_path = "datasets/ml-20m/"
    # use the last 30 days as test and anything prior as train dataset 
    spit_date = "2015-03-01 06:40:02.000000"
    train_df, test_df = load_data(fn, data_path, spit_date)
    # Movielese data is explicit so make it implicit!
    min_rating = 4.0
    model, users, items, ratings = train_als(train_df, test_df, min_rating)
    # Generate ground truth
    user_labels = dataset.generate_true_labels(test_df)
    # generate a map from train username to their user ids
    train_users_dict = dict(enumerate(users.flatten(), 0))
    training_ds = items, users, ratings
    user_recommendations = generate_user_recommendations(model, user_labels, train_users_dict, training_ds)
    user_stats = metrics.compute_basic_stats(user_recommendations, user_labels)
    print("No of users with recos: {} & no of cold users: {}".format(user_stats['users_w_recos'], user_stats['cold_start_users']))
    # sweep k and get avg p@k and recall@k, MAP, and nDCG
    als_perf = {}
    for k in range(10, 11):
        als_perf[k] = metrics.compute_precision_and_recall_at_k(user_recommendations, user_labels, k)
        print("Avergae P@{}: {}% & average Recall@{}: {}%".format(k, round(100*als_perf[k]['avg_p_at_k'], 2), 
                k, round(100*als_perf[k]['avg_recall_at_k'], 2)))
        # map_k = metrics.compute_mean_average_precision(user_recommendations, user_labels, k)
        # print("MAP@{}: {}%".format(k, round(100*map_k, 2)))


if __name__ == "__main__":
    run()
