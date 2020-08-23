from datetime import datetime

from src.data import load, split
from src.metrics import compute_p_at_k


def main():
    filename = 'historical_data.tsv'
    df = load(filename)
    split_time = datetime.strptime('2020-08-03', '%Y-%m-%d')
    train, test = split(df, split_time)
    print(train.head(10))
    print('---------------')
    print(test.head(10))
    recs = {10: [211, 212], 11: [203, 204], 20: [203, 206], 21: [], 30: []}
    k = 2
    p_at_k = compute_p_at_k(recs, test, k)
    print('P@k: {}'.format(p_at_k))


main()
