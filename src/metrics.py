import tqdm


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
