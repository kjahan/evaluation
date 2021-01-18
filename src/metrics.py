import tqdm


def _compute_p_at_k(recs, labels, k):
    """
    Given a list of recs for a user, compute P@k when user has seen items in labels.
    We assume the recs is sorted by item relevance.
    """
    return len(set(recs[:k]) & set(labels))/k


def compute_p_at_k(recommendations, test_split, k=10):
    """
    recommendations: a dictionary for recommendations where keys are user_id and values are list of recommended items
    test_split: dataframe
    return: average P@k for all users
    """
    # pre-generate all true labels for all users
    user_labels = {}
    for _, row in test_split.iterrows():
        user_id, item_id = row['user_id'], row['item_id']
        try:
            user_labels[user_id].append(item_id)
        except KeyError:
            user_labels[user_id] = [item_id]
    avg_p_at_k = 0.0
    for user_id, recs in recommendations.items():
        avg_p_at_k += _compute_p_at_k(recs, user_labels[user_id], k)
    return avg_p_at_k/len(recommendations.keys())


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
    recommendations: a dictionary for recommendations where keys are user_id and values are list of recommended items
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
