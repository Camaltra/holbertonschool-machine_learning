#!/usr/bin/env python3


"""useless comment"""

import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Bayesian Optimization Class
    """

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True,
    ):
        """
        Init the class
        :param f: The blakc-box function
        :param X_init: The dataset inital
        :param Y_init: The value of the initial
                       dataset after being process by f
        :param bounds: The bounds of the space in
                       which to look for the optimal point
        :param ac_samples: The number of samples
                           that should by analyzed during acquisition
        :param l: The length parameter for the kernel
        :param sigma_f: The std given to the output
                        of the black_box function f
        :param xsi: The exploration-exploitation factor
                    for acquisition
        :param minimize: Determining whether optimization should
                         be performed for minimization or maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(
            bounds[0],
            bounds[1],
            ac_samples
        ).reshape((-1, 1))
        self.xsi = xsi
        self.minimize = minimize
