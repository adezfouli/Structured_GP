from scipy.misc import logsumexp
from scipy.special._ufuncs import gammaln
from util import cross_ent_normal

__author__ = 'AT'

import math
from numpy.core.umath_tests import inner1d
from numpy.linalg import inv, det
import numpy as np
from GPy.util.linalg import mdot


class Likelihood:

    def __init__(self):
        pass

    def ll(self, f, y):
        raise Exception("not implemented yet")

    def ll_grad(self, f, y):
        raise Exception("not implemented yet")

    def get_num_params(self):
        raise Exception("not implemented yet")

    def set_params(self, p):
        raise Exception("not implemented yet")

    def get_params(self):
        raise Exception("not implemented yet")

    def predict(self, mu, sigma):
        raise Exception("not implemented yet")

    def ell(self, mu, sigma, Y):
        raise Exception("not implemented yet")

class MultivariateGaussian(Likelihood):
    def __init__(self, sigma):
        Likelihood.__init__(self)
        self.sigma = sigma
        self.sigma_inv = inv(self.sigma)
        self.const = -1.0 / 2 * np.log(det(self.sigma)) - float(len(self.sigma)) / 2 * np.log(2 * math.pi)

    def ll(self, f, y):
        return self.const + -1.0 / 2 * inner1d(mdot((f - y), self.sigma_inv), (f-y))

    def ll_grad(self, f, y):
        raise Exception("gradients not supported for multivariate Gaussian")

    def get_sigma(self):
        return self.sigma

    def get_params(self):
        return self.sigma.flatten()

    def get_num_params(self):
        return self.sigma.flatten().shape[0]

    def ell(self, mu, sigma, Y):
        return cross_ent_normal(mu, np.diag(sigma), Y, np.array(self.sigma))


class UnivariateGaussian(Likelihood):
    def __init__(self, sigma):
        Likelihood.__init__(self)
        self.set_params(np.log([sigma]))

    def ll(self, f, y):
        return self.const + -1.0 / 2 * inner1d(f-y, f-y) / self.sigma

    def ll_grad(self, f, y):
        return self.const_grad * self.sigma + 1.0 / 2 * inner1d(f-y, f-y) / self.sigma

    def set_params(self, p):
        self.sigma = math.exp(p[0])
        self.const = -1.0 / 2 * np.log(self.sigma) - 1 / 2 * np.log(2 * math.pi)
        self.const_grad = -1.0 / 2 / self.sigma

    def get_sigma(self):
        return np.array([[self.sigma]])

    def get_params(self):
        return np.array(np.log([self.sigma]))

    def get_num_params(self):
        return 1

    def predict(self, mu, sigma):
        return mu, sigma + self.sigma

    def ell(self, mu, sigma, Y):
        return cross_ent_normal(mu, np.diag(sigma), Y, np.array([[self.sigma]]))




class LogGaussianCox(Likelihood):
    """
    Log Gaussian Cox process

    p(y|f) = (lambda)^y exp(-lambda) / y!

    lambda = f + offset
    """

    def __init__(self, offset):
        Likelihood.__init__(self)
        self.offset = offset

    def ll(self, f, y):
        _log_lambda = (f + self.offset)
        return y * _log_lambda - np.exp(_log_lambda) - gammaln(y + 1)

    def ll_grad(self, f, y):
        return y-np.exp(f+self.offset)

    def set_params(self, p):
        self.offset = p[0]

    def get_params(self):
        return np.array([self.offset])

    def get_num_params(self):
        return 1


class LogisticLL(Likelihood):
    """
    Logistic likelihood

    p(y|f) = 1 / (1 + exp(-f))

    lambda = f + offset
    """

    def __init__(self):
        Likelihood.__init__(self)
        self.n_samples = 20000
        self.normal_samples = np.random.normal(0, 1, self.n_samples).reshape((1, self.n_samples))

    def ll(self, f, y):
        if y == 1:
            return (-(f + np.abs(f)) / 2 - np.log(1 + np.exp(-np.abs(f))))[:,0]
        if y == -1:
            return (-(-f + np.abs(-f)) / 2 - np.log(1 + np.exp(-np.abs(-f))))[:,0]

    def ll_grad(self, f, y):
        raise Exception("gradients not supported for logistic regression")

    def set_params(self, p):
        if p.shape[0] != 0:
            raise Exception("Logistic function does not have free parameters")

    def predict(self, mu, sigma):
        f = self.normal_samples * math.sqrt(sigma) + mu
        mean = np.exp(self.ll(f.T, 1)).mean()
        return mean, mean * (1 - mean)

    def get_params(self):
        return np.array([])

    def get_num_params(self):
        return 0

class SoftmaxLL(Likelihood):
    """
    Softmax likelihood:

    p(y=c|f) = exp(f_c) / (f_1 + ... + f_N)
    """

    def __init__(self):
        Likelihood.__init__(self)

    def ll(self, f, y):
        for j in range(f.shape[1]):
            f[:,j] -= f[:, y[0]]

        return -logsumexp(f, 1)

    def ll_grad(self, f, y):
        raise Exception("gradients not supported for multivariate Gaussian")

    def set_params(self, p):
        if p.shape[0] != 0:
            raise Exception("Softmax function does not have free parameters")

    def get_params(self):
        return np.array([])

    def get_num_params(self):
        return 0