# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np
from ...util.linalg import tdot
from ...util.misc import fast_array_equal
from scipy import weave
from ...util.config import *

class Linear(Kernpart):
    """
    Linear kernel

    .. math::

       k(x,y) = \sum_{i=1}^input_dim \sigma^2_i x_iy_i

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variances: the vector of variances :math:`\sigma^2_i`
    :type variances: array or list of the appropriate size (or float if there is only one variance parameter)
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel has only one variance parameter \sigma^2, otherwise there is one variance parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object
    """

    def __init__(self, input_dim, variances=None, ARD=False):
        self.input_dim = input_dim
        self.ARD = ARD
        if ARD == False:
            self.num_params = 1
            self.name = 'linear'
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == 1, "Only one variance needed for non-ARD kernel"
            else:
                variances = np.ones(1)
            self._Xcache, self._X2cache = np.empty(shape=(2,))
        else:
            self.num_params = self.input_dim
            self.name = 'linear'
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == self.input_dim, "bad number of lengthscales"
            else:
                variances = np.ones(self.input_dim)
        self._set_params(variances.flatten())

        # initialize cache
        self._Z, self._mu, self._S = np.empty(shape=(3, 1))
        self._X, self._X2, self._params = np.empty(shape=(3, 1))

        # a set of optional args to pass to weave
        weave_options_openmp = {'headers'           : ['<omp.h>'],
                                'extra_compile_args': ['-fopenmp -O3'],
                                'extra_link_args'   : ['-lgomp'],
                                'libraries': ['gomp']}
        weave_options_noopenmp = {'extra_compile_args': ['-O3']}


        if config.getboolean('parallel', 'openmp'):
            self.weave_options = weave_options_openmp
            self.weave_support_code =  """
            #include <omp.h>
            #include <math.h>
            """
        else:
            self.weave_options = weave_options_noopenmp
            self.weave_support_code = """
            #include <math.h>
            """

    def _get_params(self):
        return self.variances

    def _set_params(self, x):
        assert x.size == (self.num_params)
        self.variances = x
        self.variances2 = np.square(self.variances)

    def _get_param_names(self):
        if self.num_params == 1:
            return ['variance']
        else:
            return ['variance_%i' % i for i in range(self.variances.size)]

    def K(self, X, X2, target):
        if self.ARD:
            XX = X * np.sqrt(self.variances)
            if X2 is None:
                target += tdot(XX)
            else:
                XX2 = X2 * np.sqrt(self.variances)
                target += np.dot(XX, XX2.T)
        else:
            self._K_computations(X, X2)
            target += self.variances * self._dot_product

    def Kdiag(self, X, target):
        np.add(target, np.sum(self.variances * np.square(X), -1), target)

    def dK_dtheta(self, dL_dK, X, X2, target):
        if self.ARD:
            if X2 is None:
                [np.add(target[i:i + 1], np.sum(dL_dK * tdot(X[:, i:i + 1])), target[i:i + 1]) for i in range(self.input_dim)]
            else:
                product = X[:, None, :] * X2[None, :, :]
                target += (dL_dK[:, :, None] * product).sum(0).sum(0)
        else:
            self._K_computations(X, X2)
            target += np.sum(self._dot_product * dL_dK)

    def dKdiag_dtheta(self, dL_dKdiag, X, target):
        tmp = dL_dKdiag[:, None] * X ** 2
        if self.ARD:
            target += tmp.sum(0)
        else:
            target += tmp.sum()

    def dK_dX(self, dL_dK, X, X2, target):
        if X2 is None:
            target += 2*(((X[None,:, :] * self.variances)) * dL_dK[:, :, None]).sum(1)
        else:
            target += (((X2[None,:, :] * self.variances)) * dL_dK[:, :, None]).sum(1)

    def dKdiag_dX(self,dL_dKdiag,X,target):
        target += 2.*self.variances*dL_dKdiag[:,None]*X

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        target += np.sum(self.variances * self.mu2_S, 1)

    def dpsi0_dtheta(self, dL_dpsi0, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        tmp = dL_dpsi0[:, None] * self.mu2_S
        if self.ARD:
            target += tmp.sum(0)
        else:
            target += tmp.sum()

    def dpsi0_dmuS(self, dL_dpsi0, Z, mu, S, target_mu, target_S):
        target_mu += dL_dpsi0[:, None] * (2.0 * mu * self.variances)
        target_S += dL_dpsi0[:, None] * self.variances

    def psi1(self, Z, mu, S, target):
        """the variance, it does nothing"""
        self._psi1 = self.K(mu, Z, target)

    def dpsi1_dtheta(self, dL_dpsi1, Z, mu, S, target):
        """the variance, it does nothing"""
        self.dK_dtheta(dL_dpsi1, mu, Z, target)

    def dpsi1_dmuS(self, dL_dpsi1, Z, mu, S, target_mu, target_S):
        """Do nothing for S, it does not affect psi1"""
        self._psi_computations(Z, mu, S)
        target_mu += (dL_dpsi1[:, :, None] * (Z * self.variances)).sum(1)

    def dpsi1_dZ(self, dL_dpsi1, Z, mu, S, target):
        self.dK_dX(dL_dpsi1.T, Z, mu, target)

    def psi2(self, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        target += self._psi2

    def psi2_new(self,Z,mu,S,target):
        tmp = np.zeros((mu.shape[0], Z.shape[0]))
        self.K(mu,Z,tmp)
        target += tmp[:,:,None]*tmp[:,None,:] + np.sum(S[:,None,None,:]*self.variances**2*Z[None,:,None,:]*Z[None,None,:,:],-1)

    def dpsi2_dtheta_new(self, dL_dpsi2, Z, mu, S, target):
        tmp = np.zeros((mu.shape[0], Z.shape[0]))
        self.K(mu,Z,tmp)
        self.dK_dtheta(2.*np.sum(dL_dpsi2*tmp[:,None,:],2),mu,Z,target)
        result= 2.*(dL_dpsi2[:,:,:,None]*S[:,None,None,:]*self.variances*Z[None,:,None,:]*Z[None,None,:,:]).sum(0).sum(0).sum(0)
        if self.ARD:
            target += result.sum(0).sum(0).sum(0)
        else:
            target += result.sum()

    def dpsi2_dtheta(self, dL_dpsi2, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        tmp = dL_dpsi2[:, :, :, None] * (self.ZAinner[:, :, None, :] * (2 * Z)[None, None, :, :])
        if self.ARD:
            target += tmp.sum(0).sum(0).sum(0)
        else:
            target += tmp.sum()

    def dpsi2_dmuS_new(self, dL_dpsi2, Z, mu, S, target_mu, target_S):
        tmp = np.zeros((mu.shape[0], Z.shape[0]))
        self.K(mu,Z,tmp)
        self.dK_dX(2.*np.sum(dL_dpsi2*tmp[:,None,:],2),mu,Z,target_mu)

        Zs = Z*self.variances
        Zs_sq = Zs[:,None,:]*Zs[None,:,:]
        target_S += (dL_dpsi2[:,:,:,None]*Zs_sq[None,:,:,:]).sum(1).sum(1)

    def dpsi2_dmuS(self, dL_dpsi2, Z, mu, S, target_mu, target_S):
        """Think N,num_inducing,num_inducing,input_dim """
        self._psi_computations(Z, mu, S)
        AZZA = self.ZA.T[:, None, :, None] * self.ZA[None, :, None, :]
        AZZA = AZZA + AZZA.swapaxes(1, 2)
        AZZA_2 = AZZA/2.
        #muAZZA = np.tensordot(mu,AZZA,(-1,0))
        #target_mu_dummy, target_S_dummy = np.zeros_like(target_mu), np.zeros_like(target_S)
        #target_mu_dummy += (dL_dpsi2[:, :, :, None] * muAZZA).sum(1).sum(1)
        #target_S_dummy += (dL_dpsi2[:, :, :, None] * self.ZA[None, :, None, :] * self.ZA[None, None, :, :]).sum(1).sum(1)


        if config.getboolean('parallel', 'openmp'):
            pragma_string = "#pragma omp parallel for private(m,mm,q,qq,factor,tmp)"
        else:
            pragma_string = ''

        #Using weave, we can exploiut the symmetry of this problem:
        code = """
        int n, m, mm,q,qq;
        double factor,tmp;
        %s
        for(n=0;n<N;n++){
          for(m=0;m<num_inducing;m++){
            for(mm=0;mm<=m;mm++){
              //add in a factor of 2 for the off-diagonal terms (and then count them only once)
              if(m==mm)
                factor = dL_dpsi2(n,m,mm);
              else
                factor = 2.0*dL_dpsi2(n,m,mm);

              for(q=0;q<input_dim;q++){

                //take the dot product of mu[n,:] and AZZA[:,m,mm,q] TODO: blas!
                tmp = 0.0;
                for(qq=0;qq<input_dim;qq++){
                  tmp += mu(n,qq)*AZZA(qq,m,mm,q);
                }

                target_mu(n,q) += factor*tmp;
                target_S(n,q) += factor*AZZA_2(q,m,mm,q);
              }
            }
          }
        }
        """ % pragma_string


        N,num_inducing,input_dim = int(mu.shape[0]),int(Z.shape[0]),int(mu.shape[1])
        weave.inline(code, support_code=self.weave_support_code,
                    arg_names=['N','num_inducing','input_dim','mu','AZZA','AZZA_2','target_mu','target_S','dL_dpsi2'],
                    type_converters=weave.converters.blitz,**self.weave_options)


    def dpsi2_dZ(self, dL_dpsi2, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        #psi2_dZ = dL_dpsi2[:, :, :, None] * self.variances * self.ZAinner[:, :, None, :]
        #dummy_target = np.zeros_like(target)
        #dummy_target += psi2_dZ.sum(0).sum(0)

        AZA = self.variances*self.ZAinner

        if config.getboolean('parallel', 'openmp'):
            pragma_string = '#pragma omp parallel for private(n,mm,q)'
        else:
            pragma_string = ''

        code="""
        int n,m,mm,q;
        %s
        for(m=0;m<num_inducing;m++){
          for(q=0;q<input_dim;q++){
            for(mm=0;mm<num_inducing;mm++){
              for(n=0;n<N;n++){
                target(m,q) += dL_dpsi2(n,m,mm)*AZA(n,mm,q);
              }
            }
          }
        }
        """ % pragma_string


        N,num_inducing,input_dim = int(mu.shape[0]),int(Z.shape[0]),int(mu.shape[1])
        weave.inline(code, support_code=self.weave_support_code, 
                     arg_names=['N','num_inducing','input_dim','AZA','target','dL_dpsi2'],
                     type_converters=weave.converters.blitz,**self.weave_options)


    #---------------------------------------#
    #            Precomputations            #
    #---------------------------------------#

    def _K_computations(self, X, X2):
        if not (fast_array_equal(X, self._Xcache) and fast_array_equal(X2, self._X2cache)):
            self._Xcache = X.copy()
            if X2 is None:
                self._dot_product = tdot(X)
                self._X2cache = None
            else:
                self._X2cache = X2.copy()
                self._dot_product = np.dot(X, X2.T)

    def _psi_computations(self, Z, mu, S):
        # here are the "statistics" for psi1 and psi2
        Zv_changed = not (fast_array_equal(Z, self._Z) and fast_array_equal(self.variances, self._variances))
        muS_changed = not (fast_array_equal(mu, self._mu) and fast_array_equal(S, self._S))
        if Zv_changed:
            # Z has changed, compute Z specific stuff
            # self.ZZ = Z[:,None,:]*Z[None,:,:] # num_inducing,num_inducing,input_dim
#             self.ZZ = np.empty((Z.shape[0], Z.shape[0], Z.shape[1]), order='F')
#             [tdot(Z[:, i:i + 1], self.ZZ[:, :, i].T) for i in xrange(Z.shape[1])]
            self.ZA = Z * self.variances
            self._Z = Z.copy()
            self._variances = self.variances.copy()
        if muS_changed:
            self.mu2_S = np.square(mu) + S
            self.inner = (mu[:, None, :] * mu[:, :, None])
            diag_indices = np.diag_indices(mu.shape[1], 2)
            self.inner[:, diag_indices[0], diag_indices[1]] += S
            self._mu, self._S = mu.copy(), S.copy()
        if Zv_changed or muS_changed:
            self.ZAinner = np.dot(self.ZA, self.inner).swapaxes(0, 1)  # NOTE: self.ZAinner \in [num_inducing x N x input_dim]!
            self._psi2 = np.dot(self.ZAinner, self.ZA.T)
