from copy import deepcopy, copy
import warnings
from numpy.ma import trace
from scipy.linalg import inv, det
from sklearn import preprocessing
import GPy
from matplotlib.pyplot import show
from GPy.util.linalg import mdot
import numpy as np
from data_source import DataSource
from gsavigp import GSAVIGP
from gsavigp_single_comp import GSAVIGP_SignleComponenet
from optimizer import *
from savigp import Configuration
from cond_likelihood import multivariate_likelihood
from grad_checker import GradChecker
from plot import plot_fit
from util import chol_grad, jitchol, bcolors


class SAVIGP_Prediction:
    def __init__(self):
        pass

    @staticmethod
    def gpy_prediction(X, Y, vairiance, kernel):
        m = GPy.core.GP(X, Y, kernel=kernel, likelihood=GPy.likelihoods.Gaussian(None, vairiance))
        return m

    @staticmethod
    def prediction():
        np.random.seed(12000)
        num_input_samples = 100
        num_samples = 10000
        gaussian_sigma = 0.2

        X, Y, kernel = DataSource.normal_generate_samples(num_input_samples, gaussian_sigma)

        try:
            # for diagonal covariance
            s1 = GSAVIGP(X, Y, num_input_samples, 5, multivariate_likelihood(np.array([[gaussian_sigma]])),
                         np.array([[gaussian_sigma]]),
                         [kernel], num_samples, [
                    Configuration.MoG,
                    Configuration.ETNROPY,
                    Configuration.CROSS,
                    Configuration.ELL,
                    # Configuration.HYPER
                ])

            # for full gaussian with single component
            # s1 = GSAVIGP_SignleComponenet(X, Y, num_input_samples, multivariate_likelihood(np.array([[gaussian_sigma]])), np.array([[gaussian_sigma]]),
            # [kernel], num_samples, [
            # Configuration.MoG,
            # Configuration.ETNROPY,
            #                                         Configuration.CROSS,
            #                                         Configuration.ELL,
            #                                         # Configuration.HYPER
            #     ])

            # Optimizer.SGD(s1, 1e-16,  s1._get_params(), 2000, verbose=False, adaptive_alpha=False)
            Optimizer.BFGS(s1, max_fun=100000)
        except KeyboardInterrupt:
            pass
        print 'parameters:', s1.get_params()
        print 'num_input_samples', num_input_samples
        print 'num_samples', num_samples
        print 'gaussian sigma', gaussian_sigma
        print s1.__class__.__name__
        plot_fit(s1, plot_raw=True)

        gp = SAVIGP_Prediction.gpy_prediction(X, Y, gaussian_sigma, kernel)
        gp.plot()
        show(block=True)

if __name__ == '__main__':
    try:
        SAVIGP_Prediction.prediction()
    finally:
        pass
