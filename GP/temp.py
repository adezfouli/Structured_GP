from DerApproximator import check_d1, get_d1
from GPy.util.linalg import mdot
from numpy.linalg import inv

__author__ = 'AT'

import numpy as np

N = 10
M = 9

A = np.random.normal(-5, 5, M * N).reshape((N, M))

a = np.random.normal(1, 1, N * 1).reshape((N, 1))


def f(x):
    S = x.reshape((M, M))
    return mdot(a.T, inv(mdot(A, S, A.T)), a)[0,0]


def grad(x):
    S = x.reshape((M, M))
    return -mdot(A.T, inv(mdot(A, S, A.T)).T, a, a.T, inv(mdot(A, S, A.T)).T, A).flatten()

x1 = np.random.normal(-0.1, 20, M * M)
x1 = x1.reshape(M, M)
x1 = x1 + x1.T
x1 = x1.flatten()
aa =  get_d1(f, x1)
bb = grad(x1)

print inv(mdot(A, A.T))

# print aa
# print bb
# print sum(abs(aa - bb))


# N = 10
# M = 5
#
# A = np.random.normal(0, 1, M * N).reshape((M, N))
#
# a = np.random.normal(0, 1, N * 1).reshape((N, 1))
#
#
# def f(x):
#     S = np.zeros((M, M))
#     S[np.triu_indices(M)] = x
#     S[np.tril_indices(M)] = x
#     S = S - np.diag(S.diagonal())
#     return mdot(a.T, inv((mdot(A.T, S, A))), a)[0,0]
#
#
# def grad(x):
#     S = np.zeros((M, M))
#     S[np.triu_indices(M)] = x
#     S[np.tril_indices(M)] = x
#     S = S - np.diag(S.diagonal())
#     return mdot(A, inv((mdot(A.T, S, A))), a, a.T, inv((mdot(A.T, S, A))), A.T)
#
# x1 = np.random.normal(0, 1, (M * M - M) / 2 + M)
# print get_d1(f, x1)
# print grad(x1)
#
# print np.triu(np.array([1,2,3]).T)