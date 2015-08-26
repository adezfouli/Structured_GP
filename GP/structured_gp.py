__author__ = 'AT'

from savigp import Configuration
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
        self.bin_m = np.zeros(likelihood.bin_dim)
        self.bin_s = np.ones(likelihood.bin_dim)
        self.bin_noise = 0.1
        self.bin_kernel = np.eye(likelihood.bin_dim) * self.bin_noise
        # logger.debug("bin noise: " + str(self.bin_noise))
        np.random.seed(12000)

        self.binary_normal_samples = np.random.normal(0, 1, n_samples * likelihood.bin_dim) \
            .reshape((likelihood.bin_dim, n_samples))

        super(StructureGP, self).__init__(X, Y, num_inducing, likelihood,
                                          kernels, n_samples, config_list, latent_noise,
                                          is_exact_ell, inducing_on_Xs, n_threads, image, partition_size)

    def _update_bin_grad(self, F_B, sum_ll):
        self.dbin_m_ell = (((F_B - self.bin_m)/self.bin_s).T * sum_ll).mean(axis=1)
        self.dbin_s_ell = ((np.square((F_B - self.bin_m) / self.bin_s) - 1. / self.bin_s).T * sum_ll).mean(axis=1) / 2

    def _Kdiag(self, p_X, K, A, j, seq_poses):
        """
        calculating diagonal terms of K_tilda for latent process j (eq 4)
        """
        pxs = np.split(p_X, seq_poses[1:-1], axis=0)
        return block_diag(*tuple([self.kernels_latent[j].K(psx_i) for psx_i in pxs])) - mdot(A, K)


    def _get_A_K(self, p_X, seq_poses):
        A = np.empty((self.num_latent_proc, len(p_X), self.num_inducing))
        K = np.empty((self.num_latent_proc, len(p_X), len(p_X)))
        Kzx = np.empty((self.num_latent_proc, self.num_inducing, p_X.shape[0]))
        for j in range(self.num_latent_proc):
            Kzx[j, :, :] = self.kernels_latent[j].K(self.Z[j, :, :], p_X)
            A[j] = self._A(j, Kzx[j, :, :])
            K[j] = self._Kdiag(p_X, Kzx[j, :, :], A[j], j, seq_poses)
        return A, Kzx, K


    def _parition_ell(self, X, Y):
        """
        calculating expected log-likelihood, and it's derivatives
        :returns ell, normal ell, dell / dm, dell / ds, dell/dpi
        """

        # print 'ell started'
        total_ell = self.cached_ell
        d_ell_dm = np.zeros((self.num_mog_comp, self.num_latent_proc, self.num_inducing))
        d_ell_ds = np.zeros((self.num_mog_comp, self.num_latent_proc) + self.MoG.S_dim())
        d_ell_dPi = np.zeros(self.num_mog_comp)
        if Configuration.HYPER in self.config_list:
            d_ell_d_hyper = np.zeros((self.num_latent_proc, self.num_hyper_params))
        else:
            d_ell_d_hyper = 0

        if Configuration.LL in self.config_list:
            d_ell_d_ll = np.zeros(self.num_like_params)
        else:
            d_ell_d_ll = 0

        # self.rand_init_mog()
        if Configuration.MoG in self.config_list or \
            Configuration.LL in self.config_list or \
            self.cached_ell is None or \
            self.calculate_dhyper():
            total_ell = 0
            if self.A_cached is None:
                self.A_cached, self.Kzx_cached, self.K_cached = self._get_A_K(X, self.seq_poses)
            A = self.A_cached
            Kzx = self.Kzx_cached
            K = self.K_cached
            mean_kj = np.empty((self.num_mog_comp, self.num_latent_proc, X.shape[0]))
            sigma_kj = np.empty((self.num_mog_comp, self.num_latent_proc, X.shape[0], X.shape[0]))
            chol_sigma = np.empty((self.num_mog_comp, self.num_latent_proc, X.shape[0], X.shape[0]))
            chol_sigma_inv = np.empty((self.num_mog_comp, self.num_latent_proc, X.shape[0], X.shape[0]))
            sigma_inv = np.empty((self.num_mog_comp, self.num_latent_proc, X.shape[0], X.shape[0]))
            F = np.empty((self.n_samples, X.shape[0], self.num_latent_proc))
            for k in range(self.num_mog_comp):
                for j in range(self.num_latent_proc):
                    norm_samples = self.normal_samples[j, :, :X.shape[0]]
                    mean_kj[k,j] = self._b(k, j, A[j], Kzx[j].T)
                    sigma_kj[k,j] = self._sigma(k, j, K[j], A[j], Kzx[j].T)
                    chol_sigma[k,j], chol_sigma_inv[k,j], sigma_inv[k,j] = self._chol_sigma(sigma_kj[k,j])
                    F[:, :, j] = mdot(norm_samples, chol_sigma[k,j].T)
                    F[:, :, j] = F[:, :, j] + mean_kj[k,j]
                F_B = self.get_binary_sample()
                cond_ll, grad_ll, total_ell, sum_ll = self.cond_likelihood.ll_F_Y(F, Y, F_B)
                self._update_bin_grad(F_B, sum_ll)
                for j in range(self.num_latent_proc):
                    norm_samples = self.normal_samples[j, :, :X.shape[0]]
                    sfb = mdot(norm_samples, chol_sigma_inv[k,j]) # sigma^ -1 (f - b)
                    m = self._average(cond_ll, sfb, True)
                    d_ell_dm[k,j] = self._proj_m_grad(j, mdot(m, Kzx[j].T)) * self.MoG.pi[k]
                    from structured_gp import StructureGP
                    d_ell_ds[k,j] = StructureGP._struct_dell_ds(self, k, j, cond_ll, A, sigma_inv[k,j], sfb)
                    if self.calculate_dhyper():
                        ds_dhyp = self._dsigma_dhyp(j, k, A[j], Kzx, X)
                        db_dhyp = self._db_dhyp(j, k, A[j], X)
                        for h in range(self.num_hyper_params):
                            d_ell_d_hyper[j, h] += -1./2 * self.MoG.pi[k] * (
                                                self._average(cond_ll,
                                                np.ones(cond_ll.shape) / sigma_kj[k, j] * ds_dhyp[:, h] +
                                                -2. * norm_samples / np.sqrt(sigma_kj[k,j]) * db_dhyp[:, h]
                                                - np.square(norm_samples)/sigma_kj[k, j] * ds_dhyp[:, h], True)).sum()

                sum_cond_ll = cond_ll.sum() / self.n_samples
                d_ell_dPi[k] = sum_cond_ll

                if Configuration.LL in self.config_list:
                    d_ell_d_ll += self.MoG.pi[k] * grad_ll.sum() / self.n_samples

            if self.is_exact_ell:
                total_ell = 0
                for n in range(len(X)):
                    for k in range(self.num_mog_comp):
                            total_ell += self.cond_likelihood.ell(np.array(mean_kj[k, :, n]), np.array(sigma_kj[k, :, n]), Y[n, :]) * self.MoG.pi[k]

        return total_ell, d_ell_dm, d_ell_ds, d_ell_dPi, d_ell_d_hyper, d_ell_d_ll, None


    def rand_init_mog(self):
        super(StructureGP, self).rand_init_mog()
        self.bin_m = np.random.uniform(low=.1, high=100., size=self.bin_m.shape[0])
        self.bin_s = np.random.uniform(low=1, high=1, size=self.bin_s.shape[0])

    def _sigma(self, k, j, Kj, Aj, Kzx):
        return Kj + self.MoG.aSa_full(Aj, k, j)

    def _chol_sigma(self, sigma, seq_poses = None):
        if seq_poses is None:
            seq_poses = self.seq_poses

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
        bin_dM = np.zeros(self.bin_dM.shape)
        bin_dL = np.zeros(self.bin_dL.shape)
        if Configuration.CROSS in self.config_list and Configuration.ENTROPY in self.config_list:
            self.ll += self.bin_ent + self.bin_NCE
            bin_dM += self.bin_dM
            bin_dL += self.bin_dL * self.bin_s
        if Configuration.ELL in self.config_list:
            bin_dM += self.dbin_m_ell
            bin_dL += self.dbin_s_ell * self.bin_s

        self.grad_ll = np.hstack((self.grad_ll, bin_dM, bin_dL))

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

    def set_all_params(self, p):
        self.bin_s = np.exp(p[-self.bin_m.shape[0]:])
        self.bin_m = p[-(self.bin_m.shape[0]+self.bin_s.shape[0]):-self.bin_m.shape[0]].copy()
        super(StructureGP, self).set_all_params(p[:-(self.bin_m.shape[0]+self.bin_s.shape[0])])

    def get_params(self):
        parent_params = super(StructureGP, self).get_params()
        return np.hstack((parent_params, self.bin_m.copy(), np.log(self.bin_s.copy())))

    def set_params(self, p):
        self.bin_s = np.exp(p[-self.bin_m.shape[0]:])
        self.bin_m = p[-(self.bin_m.shape[0]+self.bin_s.shape[0]):-self.bin_m.shape[0]].copy()
        super(StructureGP, self).set_params(p[:-(self.bin_m.shape[0]+self.bin_s.shape[0])])

    def get_param_names(self):
        parent_names = super(StructureGP, self).get_param_names()
        return parent_names + ['b_m'] * self.cond_likelihood.bin_dim + ['b_s'] * self.cond_likelihood.bin_dim

    def _predict_comp(self, Xs, Ys):
        A, Kzx, K = self._get_A_K(Xs, self.cond_likelihood.test_seq_poses)

        predicted_mu = np.empty((Xs.shape[0], self.num_mog_comp, self.cond_likelihood.output_dim()))
        predicted_var = np.empty((Xs.shape[0], self.num_mog_comp, self.cond_likelihood.output_dim()))
        nlpd = None
        if not (Ys is None):
            nlpd = np.empty((Xs.shape[0], self.cond_likelihood.nlpd_dim(), self.num_mog_comp))

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
