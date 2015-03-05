import math
import numpy as np
from numpy.ma import trace
from scipy import linalg
from scipy.linalg import det, inv, lapack
from GPy.util.linalg import mdot, dpotri


def mdiag_dot(A, B):
    return np.einsum('ij,ji -> i', A, B)


def KL_normal(m1, sigma1, m2, sigma2):
    return 1. / 2. * (math.log(det(sigma2) / det(sigma1)) - len(m1) + trace(mdot(inv(sigma2), sigma1)) + \
    mdot((m2 - m1).T, inv(sigma2) , m2- m1))


def cross_ent_normal(m1, sigma1, m2, sigma2):
    """
    calculates cross entropy between two Gaussian
    """
    return -KL_normal(m1, sigma1, m2, sigma2) - 1. / 2 * math.log(det(2.0 * math.pi * math.e * sigma1))

def jitchol(A, maxtries=5):
    A = np.ascontiguousarray(A)
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise linalg.LinAlgError, "not pd: non-positive diagonal elements"
        jitter = diagA.mean() * 1e-6
        while maxtries > 0 and np.isfinite(jitter):
            try:
                L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                return L
            except:
                jitter *= 10
            finally:
                maxtries -= 1
        raise linalg.LinAlgError, "not positive definite, even with jitter."

def pddet(L):
    """
    Determinant of a positive definite matrix, only symmetric matricies though
    """
    logdetA = 2*sum(np.log(np.diag(L)))
    return logdetA


def inv_chol(L):
    Ai, _ = dpotri(np.asfortranarray(L), lower=1)
    return Ai

def chol_grad(L, dM_dx):
    return mdot(dM_dx+dM_dx.T, L)

def chol_grad_with_for(L, dM_dx):
    d = L.shape[0]
    J = np.zeros((d,d))
    dM_dL = np.empty((d,d))
    for i in range(d):
        for j in range(i + 1):
            # J[j,i] = (i == j) * L[i,i] + (i != j)
            J[j,i] = 1.0
            tmp = mdot(L, J)
            dM_dL[i,j] = ((tmp + tmp.T) * (dM_dx)).sum()
            J[j,i] = 0.0
    return dM_dL

import numpy as np,numpy.linalg

def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n)
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk

def log_diag_gaussian(m1, m2, s):
    const = -1.0 / 2 * np.log(s).sum() - float(len(s)) / 2 * np.log(2 * math.pi)
    return const + -1.0 / 2 * np.dot((m1 - m2) / s, (m1-m2).T)

