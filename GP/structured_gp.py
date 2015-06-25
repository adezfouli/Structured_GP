__author__ = 'AT'

from scipy.misc import logsumexp
from GPy.util.linalg import mdot
from scipy.linalg import block_diag, inv
import numpy as np
from util import jitchol, inv_chol
from savigp_single_comp import SAVIGP_SingleComponent


class StructureGP(SAVIGP_SingleComponent):
    def __init__(self, X, Y, num_inducing, likelihood, kernels, n_samples,
                 config_list, latent_noise, is_exact_ell, inducing_on_Xs, n_threads=1, image=None, partition_size=3000, logger = None):
        self.seq_poses = likelihood.seq_poses
        self.A_cached = None
        self.bin_m = np.zeros(likelihood.bin_dim) + 1
        self.bin_s = np.ones(likelihood.bin_dim)
        self.bin_noise = 1e-4
        self.bin_kernel = np.eye(likelihood.bin_dim) * self.bin_noise
        np.random.seed(12000)

        self.binary_normal_samples = np.random.normal(0, 1, n_samples * likelihood.bin_dim) \
            .reshape((likelihood.bin_dim, n_samples))

        super(StructureGP, self).__init__(X, Y, num_inducing, likelihood,
                                          kernels, n_samples, config_list, latent_noise,
                                          is_exact_ell, inducing_on_Xs, n_threads, image, partition_size, logger)

    def rand_init_mog(self):
        super(StructureGP, self).rand_init_mog()
        self.bin_m = np.random.uniform(low=1., high=2., size=self.bin_m.shape[0])
        self.bin_s = np.random.uniform(low=1., high=3., size=self.bin_s.shape[0])

    def _partition_data(self, X, Y):
        X_paritions = []
        Y_paritions = []
        paritions = np.array_split(range(self.seq_poses.shape[0]-1), min(self.n_threads, self.seq_poses.shape[0]-1))
        max_partition_size = None
        for p in paritions:
            Y_paritions.append(p)
            X_paritions.append(self._get_X_indices(p))
            if max_partition_size is None or self.seq_poses[p[-1]+1] - self.seq_poses[p[0]] +1 > max_partition_size:
                max_partition_size = self.seq_poses[p[-1]+1] - self.seq_poses[p[0]] +1

        return X_paritions, Y_paritions, self.n_threads, max_partition_size

    def _sigma(self, k, j, Kj, Aj, Kzx):
        """
        calculating [sigma_k(n)]j,j for latent process j (eq 20) for all k
        """
        return Kj + self.MoG.aSkja_full(Aj, k, j)

    def _chol_sigma(self, sigma, seq_poses=None):
        if seq_poses is None:
            seq_poses = self.seq_poses

        seq_poses = seq_poses - seq_poses[0]
        sigmas = [0] * (len(seq_poses) - 1)
        inv_sigmas = [0] * (len(seq_poses) - 1)
        inv_sigmas_chol = [0] * (len(seq_poses) - 1)
        for s in range(len(seq_poses) - 1):
            sigmas[s] = jitchol(sigma[seq_poses[s]:seq_poses[s + 1], seq_poses[s]:seq_poses[s + 1]])
            inv_sigmas[s] = inv_chol(sigmas[s])
            inv_sigmas_chol[s] = inv(sigmas[s])
        return block_diag(*tuple(sigmas)), \
               block_diag(*tuple(inv_sigmas_chol)), \
               block_diag(*tuple(inv_sigmas))

    def _break_to_blocks(self, A):
        breaks = [0] * (len(self.seq_poses) - 1)
        for s in range(len(self.seq_poses) - 1):
            breaks[s] = A[self.seq_poses[s]:self.seq_poses[s + 1], self.seq_poses[s]:self.seq_poses[s + 1]]
        return breaks

    # @numba.jit
    def _struct_dell_ds(self, k, j, cond_ll, A, inv_sigma, sfb):
        output = np.zeros((A.shape[2], A.shape[2]))
        for s in range(len(self.seq_poses) - 1):
            sub_cond = cond_ll[:, self.seq_poses[s]:self.seq_poses[s + 1]]
            sub_inv_sigma = inv_sigma[self.seq_poses[s]:self.seq_poses[s + 1], self.seq_poses[s]:self.seq_poses[s + 1]]
            sbbs = sfb[:, self.seq_poses[s]:self.seq_poses[s + 1], np.newaxis] * \
                   sfb[:, np.newaxis, self.seq_poses[s]:self.seq_poses[s + 1]]
            tmp = self._average(np.repeat(sub_cond, sbbs.shape[2], axis=1),
                                   sbbs.reshape(sbbs.shape[0], sbbs.shape[1] * sbbs.shape[2]) -
                                   sub_inv_sigma.flatten(),
                                   True).reshape(sbbs.shape[1], sbbs.shape[2])
            output += mdot(A[j, self.seq_poses[s]:self.seq_poses[s + 1], :].T, tmp,
                           A[j, self.seq_poses[s]:self.seq_poses[s + 1], :]) * self.MoG.pi[k] / 2.

        # sbbs = sfb[:, :, np.newaxis] * sfb[:, np.newaxis, :]
        # tmp = self._average(np.repeat(cond_ll, sbbs.shape[2], axis=1),
        #                            sbbs.reshape(sbbs.shape[0], sbbs.shape[1] * sbbs.shape[2]) -
        #                            inv_sigma.flatten(),
        #                            True).reshape(sbbs.shape[1], sbbs.shape[2])
        # output2 =  mdot( A[j].T, tmp, A[j]) * self.MoG.pi[k] / 2.
        #
        # print output
        # print output2
        return output

    def _bin_KL(self):
        mu = self.bin_m
        sBin = self.bin_s
        dimS = sBin.shape[0]
        sinv = 1. / sBin
        k = np.diagonal(self.bin_kernel)
        kinv = 1. / k
        self.bin_ent = 0.5 * sum(np.log(sBin))
        self.bin_NCE = -0.5 * sum(np.log(k)) \
                  - 0.5 * np.dot((np.square(mu)), kinv) \
                  - 0.5 * np.dot(kinv, sBin) \
                  + 0.5 * dimS
        self.bin_dM = -kinv * mu
        self.bin_dL = 0.5 * (sinv - kinv)

    def _update(self):
        self._bin_KL()
        super(StructureGP, self)._update()
        self.ll += self.bin_ent + self.bin_NCE
        self.grad_ll = np.hstack((self.grad_ll, self.bin_dM, self.bin_dL * self.bin_s))

    def get_binary_sample(self, n_samples = None):
        if n_samples is None:
            samples = self.binary_normal_samples
        else:
            samples = np.random.normal(0, 1, n_samples * self.cond_likelihood.bin_dim) \
            .reshape((self.cond_likelihood.bin_dim, n_samples))

        return samples.T * np.sqrt(self.bin_s) + self.bin_m

    def get_all_params(self):
        parent_params = super(StructureGP, self).get_all_params()
        return np.hstack((parent_params, self.bin_m.copy(), np.log(self.bin_s.copy())))

    def get_params(self):
        parent_params = super(StructureGP, self).get_params()
        return np.hstack((parent_params, self.bin_m, np.log(self.bin_s)))

    def set_params(self, p):
        self.bin_s = np.exp(p[-self.bin_m.shape[0]:])
        self.bin_m = p[-(self.bin_m.shape[0]+self.bin_s.shape[0]):-self.bin_m.shape[0]]
        super(StructureGP, self).set_params(p[:-(self.bin_m.shape[0]+self.bin_s.shape[0])])

    def get_param_names(self):
        parent_names = super(StructureGP, self).get_param_names()
        return parent_names + ['b_m'] * self.cond_likelihood.bin_dim + ['b_s'] * self.cond_likelihood.bin_dim

    def _predict_comp(self, Xs, Ys):
        A, Kzx, K = self._get_A_K(Xs)

        predicted_mu = np.empty((Xs.shape[0], self.num_mog_comp, self.cond_likelihood.output_dim()))
        predicted_var = np.empty((Xs.shape[0], self.num_mog_comp, self.cond_likelihood.output_dim()))
        nlpd = None
        if not (Ys is None):
            nlpd = np.empty((Xs.shape[0], self.num_mog_comp))

        mean_kj = np.empty((self.num_mog_comp, self.num_latent_proc, Xs.shape[0]))
        sigma_kj = np.empty((self.num_mog_comp, self.num_latent_proc, Xs.shape[0], Xs.shape[0]))
        chol_sigma = np.empty((self.num_mog_comp, self.num_latent_proc, Xs.shape[0], Xs.shape[0]))
        for k in range(self.num_mog_comp):
            for j in range(self.num_latent_proc):
                mean_kj[k,j] = self._b(k, j, A[j], Kzx[j].T)
                sigma_kj[k,j] = self._sigma(k, j, K[j], A[j], Kzx[j].T)
                chol_sigma[k,j], _, _ = self._chol_sigma(sigma_kj[k,j], self.cond_likelihood.test_seq_poses)

            if not (Ys is None):
                predicted_mu[:, k, :], predicted_var[:, k, :], nlpd[:, k] = \
                    self.cond_likelihood.predict(mean_kj[k, :].T, chol_sigma[k, :].T, Ys, self)
            else:
                predicted_mu[:, k, :], predicted_var[:, k, :], _ = \
                    self.cond_likelihood.predict(mean_kj[k, :].T, sigma_kj[k, :].T, Ys, self)

        return predicted_mu, predicted_var, -logsumexp(nlpd, 1, self.MoG.pi)

