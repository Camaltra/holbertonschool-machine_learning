#!/usr/bin/env python3


"""useless comment"""


import sklearn.mixture


def gmm(X, k):
    """
    Use the GMN of scikit learn
    :param X: The data set
    :param k: The number of cluster
    :return: The prior, means, covariance, clss, bic
    """
    gmm = sklearn.mixture.GaussianMixture(k).fit(X)

    return (gmm.weights_, gmm.means_, gmm.covariances_,
            gmm.predict(X), gmm.bic(X))
