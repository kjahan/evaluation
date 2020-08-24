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
