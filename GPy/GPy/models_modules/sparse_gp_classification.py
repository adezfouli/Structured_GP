# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import SparseGP
from .. import likelihoods
from .. import kern
from ..likelihoods import likelihood

class SparseGPClassification(SparseGP):
    """
    sparse Gaussian Process model for classification

    This is a thin wrapper around the sparse_GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param likelihood: a GPy likelihood, defaults to Bernoulli with probit link_function
    :param kernel: a GPy kernel, defaults to rbf+white
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True
    :rtype: model object

    """

    def __init__(self, X, Y=None, likelihood=None, kernel=None, normalize_X=False, normalize_Y=False, Z=None, num_inducing=10):
        if kernel is None:
            kernel = kern.rbf(X.shape[1])# + kern.white(X.shape[1],1e-3)

        if likelihood is None:
            noise_model = likelihoods.bernoulli()
            likelihood = likelihoods.EP(Y, noise_model)
        elif Y is not None:
            if not all(Y.flatten() == likelihood.data.flatten()):
                raise Warning, 'likelihood.data and Y are different.'

        if Z is None:
            i = np.random.permutation(X.shape[0])[:num_inducing]
            Z = X[i].copy()
        else:
            assert Z.shape[1] == X.shape[1]

        SparseGP.__init__(self, X, likelihood, kernel, Z=Z, normalize_X=normalize_X)
        self.ensure_default_constraints()

    def getstate(self):
        return SparseGP.getstate(self)


    def setstate(self, state):
        return SparseGP.setstate(self, state)

    pass
