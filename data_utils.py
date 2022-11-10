import numpy as np

def gen_range(beg, end, interval, std_noise=0.0):
    # создаём массив с определённым интервалом
    X_base = np.arange(beg, end, interval)

    # добавляем шум к данным
    return X_base + np.random.normal(0.0, std_noise, size=X_base.shape)

def transform_feature(data, func):
    return np.apply_along_axis(func, 0, data)

def std_noise(X, std):
    return X + np.random.normal(0.0, std, size=X.shape)

def prepare_features(X, poly=1):
    if(len(X.shape) == 1):
        X = np.matrix(X).T
    else:
        X = np.matrix(X)

    X_res = np.ones((X.shape[0], 1))

    for i in range(X.shape[1]):
        # создаём признаки для каждой степени (полиноминальная регрессия, как многопеременная линейная регрессия)
        for p in range(1, poly + 1):
            # берём i-й столбец и возводим его в нужную степень
            c = np.matrix(np.power(X[:,i], p))
            # добавляем возведённый в степень массив
            X_res = np.concatenate([X_res, c], axis=-1)

    return X_res

def prepare_target(Y):
    return np.matrix(Y).T

def minmax_features(X):
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    return (X - min) / (max - min), min, max

def revert_minmax(X, min, max):
    return X * (max - min) + min

def std_scale_features(X):
    m = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - m) / std, m, std

def revert_std_scale(X, mean, std):
    return X * std + mean

def take_random(X, Y, batch_size):
    idx_batch = np.random.randint(0,X.shape[0],batch_size)
    x_batch   = np.take(X, idx_batch, axis=0)
    y_batch   = np.take(Y, idx_batch, axis=0)
    return x_batch, y_batch

def split_test_train(X, Y, train_percent):
    train_count = int(X.shape[0] * train_percent)
    indices = np.random.permutation(X.shape[0])
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]
    return X[train_indices,:], X[test_indices, :], Y[train_indices, :], Y[test_indices, :]