import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, KFold
from scipy.spatial import distance_matrix

def predict_KRR(X_train, X_test, y_train, y_test, sigma=100, l2reg=1e-6):

    g_gauss = 1.0 / (2 * sigma ** 2)

    K = rbf_kernel(X_train, X_train, gamma=g_gauss)
    K[np.diag_indices_from(K)] += l2reg
    alpha = np.dot(np.linalg.inv(K), y_train)
    K_test = rbf_kernel(X_test, X_train, gamma=g_gauss)

    y_pred = np.dot(K_test, alpha)
    mae = np.mean(np.abs(y_test - y_pred))
    return mae, y_pred


def opt_hyperparams(
    X, y, CV=5, seed=100, 
    sigmas=[0.1, 1, 10, 100, 1000, 10000], 
    l2regs=[1e-10, 1e-8, 1e-6, 1e-4],
):

    maes = np.zeros((CV, len(sigmas), len(l2regs)))

    kfold = KFold(n_splits=CV, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train_i, X_test_i = X[train_idx], X[test_idx]
        y_train_i, y_test_i = y[train_idx], y[test_idx]

        for j, sigma in enumerate(sigmas):
            for k, l2reg in enumerate(l2regs):
                mae, y_pred = predict_KRR(
                    X_train_i, X_test_i, y_train_i, y_test_i, sigma=sigma, l2reg=l2reg
                )
                maes[i, j, k] = mae

    mean_maes = np.mean(maes, axis=0)

    min_j, min_k = np.unravel_index(np.argmin(mean_maes, axis=None), mean_maes.shape)
    min_sigma = sigmas[min_j]
    min_l2reg = l2regs[min_k]

    print(
        "min mae",
        mean_maes[min_j, min_k],
        "for sigma=",
        min_sigma,
        "and l2reg=",
        min_l2reg,
    )
    return min_sigma, min_l2reg

                    

def learning_curve(X, y, CV=5, n_points=5, seed=100, sigma=1, l2reg=1e-6):

    train_fractions = np.logspace(-1, 0, num=n_points, endpoint=True)

    maes = np.zeros((CV, n_points))

    kfold = KFold(n_splits=CV, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
        print("CV iteration", i)
        X_train_i, X_test_i = X[train_idx], X[test_idx]
        y_train_i, y_test_i = y[train_idx], y[test_idx]

        train_sizes = [int(len(y_train_i)*x) for x in train_fractions]

        for j, train_size in enumerate(train_sizes):
            X_train = X_train_i[:train_size]
            y_train = y_train_i[:train_size]
            mae, _ = predict_KRR(X_train, X_test_i, 
                                y_train, y_test_i, 
                                sigma=sigma, l2reg=l2reg)
            maes[i,j] = mae

    mean_maes = np.mean(maes, axis=0)
    stdev = np.std(maes, axis=0)
    return train_sizes, mean_maes, stdev


def FPS(D, npoints):
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points
    Return
    ------
    perm : N-length array of indices
           permutation indices
    """
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = np.zeros(npoints, dtype=np.int64)
    lambdas = np.zeros(npoints)
    ds = D[0, :]
    for i in range(1, npoints):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return perm


def FPS_learning_curve(X, y, CV=10, sigma=10, l2reg=1e-10):
    train_fractions = np.logspace(-1, 0, num=5, endpoint=True)
    maes = np.zeros((CV, 5))
    kfold = KFold(n_splits=CV, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train_i, X_test_i = X[train_idx], X[test_idx]
        y_train_i, y_test_i = y[train_idx], y[test_idx]

        D_train = distance_matrix(X_train_i, X_train_i)

        train_sizes = [int(len(y_train_i) * x) for x in train_fractions]

        for j, train_size in enumerate(train_sizes):
            sub_train_indices =  FPS(D_train, train_size)
            X_train = X_train_i[sub_train_indices]
            y_train = y_train_i[sub_train_indices]
            mae, _ = predict_KRR(X_train, X_test_i, y_train, y_test_i,
                                sigma=sigma, l2reg=l2reg)
            maes[i,j] = mae
    mean_maes = np.mean(maes, axis=0)
    stdev = np.std(maes, axis=0)
    return train_sizes, mean_maes, stdev
