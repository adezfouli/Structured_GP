from GPy.util.linalg import mdot
from scipy.linalg import block_diag, cho_solve, inv
from util import jitchol, inv_chol
from savigp_single_comp import SAVIGP_SingleComponent
import numpy as np
__author__ = 'AT'

class StructureGP(SAVIGP_SingleComponent):

    def __init__(self, X, Y, num_inducing, likelihood, kernels, n_samples,
                 config_list, latent_noise, is_exact_ell, inducing_on_Xs, n_threads =1, image=None, partition_size=3000):
        self.seq_poses = likelihood.seq_poses
        self.A_cached = None
        super(StructureGP, self).__init__(X, Y, num_inducing, likelihood,
                                                     kernels, n_samples, config_list, latent_noise,
                                                     is_exact_ell, inducing_on_Xs, n_threads, image, partition_size)

    def _sigma(self, k, j, Kj, Aj, Kzx):
        """
        calculating [sigma_k(n)]j,j for latent process j (eq 20) for all k
        """
        return Kj + self.MoG.aSkja_full(Aj, k, j)


    def _chol_sigma(self, sigma):
        sigmas = [0]* (len(self.seq_poses) - 1)
        inv_sigmas = [0]* (len(self.seq_poses) - 1)
        inv_sigmas_chol = [0]* (len(self.seq_poses) - 1)
        for s in range(len(self.seq_poses) -1):
            sigmas[s] = jitchol(sigma[self.seq_poses[s]:self.seq_poses[s+1], self.seq_poses[s]:self.seq_poses[s+1]])
            inv_sigmas[s] = inv_chol(sigmas[s])
            inv_sigmas_chol[s] = inv(sigmas[s])
        return block_diag(*tuple(sigmas)), \
               block_diag(*tuple(inv_sigmas_chol)),\
               block_diag(*tuple(inv_sigmas))

    def _break_to_blocks(self, A):
        breaks = [0]*(len(self.seq_poses) - 1)
        for s in range(len(self.seq_poses) -1):
            breaks[s] = A[self.seq_poses[s]:self.seq_poses[s+1], self.seq_poses[s]:self.seq_poses[s+1]]
        return breaks

    def _struct_dell_ds(self, k, j, cond_ll, A, sigma_kj, norm_samples, inv_sigma):
        return  mdot(A[j].T * self._average(cond_ll, mdot(norm_samples**2 - np.ones((norm_samples.shape[0], 1)), inv_sigma), True), A[j]) \
                                                * self.MoG.pi[k] / 2.

