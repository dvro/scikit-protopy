import numpy as np


def random_subset(iterator, k):
    result = iterator[:k]
    i = k
    tmp_it = iterator[k:]
    for item in tmp_it:
        i = i + 1
        s = int(np.random.random() * i)
        if s < k:
            result[s] = item
    return result

def generate_imbalance(X, y, positive_label=1, ir=2):
    mask = y == positive_label
    seq = np.arange(y.shape[0])[mask]
    k = float(sum(mask))/ir
    idx = np.asarray(random_subset(seq, int(k)))
    mask = ~mask
    mask[idx] = True
    return X[mask], y[mask]



