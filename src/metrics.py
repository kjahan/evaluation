import tqdm
import math


def compute_basic_stats(recommendations, user_labels):
    """
    recommendations: a dictionary for recommendations where keys are user_id and values are list of recommended items
    user_labels: a dictionary from username to items a user has actually seen during test period
    return: no of users with recos in test fold and no of cold start users
    """
    users_no, cold_start_users = 0, 0
    with tqdm.tqdm(total=len(user_labels)) as progress:
        for username, actual_items_seen in user_labels.items():
            if username in recommendations:
                users_no += 1
            else:
                cold_start_users += 1
                continue
            progress.update(1)
    return {"users_w_recos": users_no, "cold_start_users": cold_start_users}


def compute_precision_and_recall_at_k(recommendations, user_labels, k):
    """
    recommendations: a dictionary for recommendations where keys are user_id and values are list of tuples consisting recommended items & their relevance scores
    user_labels: a dictionary from username to items a user has actually seen during test period
    k: no of recos per test user
    return: average precision and recall at k
    """
    users_no = 0
    sum_p_at_k, sum_recall_at_k = 0, 0
    with tqdm.tqdm(total=len(user_labels)) as progress:
        for username, actual_items_seen in user_labels.items():
            if username in recommendations:
                users_no += 1
                # grab only the top k recommended items for evaluation
                print(recommendations[username][:k])
                recommended_items = [item[0] for item in recommendations[username][:k]]
                # compute p@k
                sum_p_at_k += len(set(recommended_items) & set(actual_items_seen))/k
                sum_recall_at_k += len(set(recommended_items) & set(actual_items_seen))/len(actual_items_seen)
            progress.update(1)
    # compute avg p@k
    avg_p_at_k = sum_p_at_k/users_no
    # compute avg recall at k
    avg_recall_at_k = sum_recall_at_k/users_no
    return {"avg_p_at_k": avg_p_at_k, "avg_recall_at_k": avg_recall_at_k}


def compute_average_precison_for_a_query(recommended_items, actual_items_seen):
    """
    recommended_items: list of n recommended item for a test user
    actual_items_seen: list of actual items that the user has viewed
    return: average precision for a query (user/item)
    """
    num_rel_items = len(actual_items_seen)
    ave_p = 0
    for i in range(1, len(recommended_items)):
        if recommended_items[i] in set(actual_items_seen):
            p_at_i = len(set(recommended_items[:i]) & set(actual_items_seen))/i
            # item i is relevant
            ave_p += p_at_i
    ave_p = ave_p/(num_rel_items)
    return ave_p


def compute_mean_average_precision(recommendations, user_labels, n):
    """
    return: Mean Average Precision
    """
    queries_no, mean_average_precision = 0, 0
    with tqdm.tqdm(total=len(user_labels)) as progress:
        for username, actual_items_seen in user_labels.items():
            if username in recommendations:
                queries_no += 1
                # grab only the top n recommended items for evaluation
                recommended_items = [item[0] for item in recommendations[username][:n]]
                mean_average_precision += compute_average_precison_for_a_query(recommended_items, actual_items_seen)
            progress.update(1)
    # compute MAP
    mean_average_precision = mean_average_precision/queries_no
    return mean_average_precision


def compute_dcg(recommended_items, actual_items_seen):
    """
    recommended_items: list of n recommended item for a test user
    actual_items_seen: list of actual items that the user has viewed
    return: Discounted cumulative gain for a query (user/item)
    """
    dcg = 0
    for i in range(1, len(recommended_items)):
        if recommended_items[i] in set(actual_items_seen):
            # item i is relevant
            dcg += 1.0/math.log(i+1, 2)
    return dcg


def compute_normalized_dcg(recommendations, user_labels, n):
    """
    return: nDCG
    """
    queries_no, n_dcg = 0, 0
    with tqdm.tqdm(total=len(user_labels)) as progress:
        for username, actual_items_seen in user_labels.items():
            if username in recommendations:
                queries_no += 1
                # grab only the top n recommended items for evaluation
                recommended_items = [item[0] for item in recommendations[username][:n]]
                dcg = compute_dcg(recommended_items, actual_items_seen)
                # Let's compute IDCG
                intersections = list(set(recommended_items) & set(actual_items_seen))
                diff = list(set(recommended_items) - set(actual_items_seen))
                ideal_recommended_items = intersections + diff
                ideal_dcg = compute_dcg(ideal_recommended_items, actual_items_seen)
                n_dcg = dcg/ideal_dcg
            progress.update(1)
    # compute normalized DCG
    n_dcg = n_dcg/queries_no
    return n_dcg
