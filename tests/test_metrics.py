import src.metrics as metrics
import pytest


def mock_recoms():
    recommendations = {1: [(10, 0.1), (20, 0.05)], 2: [(30, 0.8), (40, 0.01)], 3: [(10, 0.5), (50, 0.35)]}
    user_labels = {1: [10, 30, 60], 2: [20, 50, 70], 3: [10, 50], 6: [10, 70]}
    return recommendations, user_labels


def test_compute_precision_and_recall_at_1():
    recommendations, user_labels = mock_recoms()
    perf = metrics.compute_precision_and_recall_at_k(recommendations, user_labels, 1)
    expected_p_at_1 = 2/3.0
    expected_recall_at_1 = 5.0/18 #((1/3.)+(1./2))/3
    assert perf['avg_p_at_k'] == pytest.approx(expected_p_at_1)
    assert perf['avg_recall_at_k'] == pytest.approx(expected_recall_at_1)


def test_compute_precision_and_recall_at_2():
    recommendations, user_labels = mock_recoms()
    perf = metrics.compute_precision_and_recall_at_k(recommendations, user_labels, 2)
    expected_p_at_2 = 0.5
    expected_recall_at_2 = 4.0/9
    assert perf['avg_p_at_k'] == pytest.approx(expected_p_at_2)
    assert perf['avg_recall_at_k'] == pytest.approx(expected_recall_at_2)
