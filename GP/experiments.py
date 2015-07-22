import logging
import math
import pickle
from GPy.kern import Linear
from ExtRBF import ExtRBF
from structured_gp import StructureGP
from gpstruct_wrapper import gpstruct_wrapper
from data_transformation import MeanTransformation, IdentityTransformation, MinTransformation
from savigp import SAVIGP
from savigp_diag import SAVIGP_Diag
from savigp_single_comp import SAVIGP_SingleComponent
import csv
import GPy
from sklearn import preprocessing
from likelihood import UnivariateGaussian, LogisticLL, SoftmaxLL, LogGaussianCox, WarpLL, StructLL, CogLL
from data_source import DataSource
import numpy as np
from optimizer import Optimizer
from util import id_generator, check_dir_exists, get_git
import Image

class Experiments:
    @staticmethod
    def get_output_path():
        return '../results/'

    @staticmethod
    def get_logger_path():
        return '../logs/'

    @staticmethod
    def get_logger(name, level):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        check_dir_exists(Experiments.get_logger_path())
        fh = logging.FileHandler(Experiments.get_logger_path() + name + '.log')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    @staticmethod
    def export_train(name, Xtrain, Ytrain, export_X=False):
        path = Experiments.get_output_path() + name + '/'
        check_dir_exists(path)
        file_name = 'train_'
        header =['Y%d,' % (j) for j in range(Ytrain.shape[1])]
        data= None
        if export_X:
            data = np.hstack((Ytrain, Xtrain))
            header += ['X%d,' % (j) for j in range(Xtrain.shape[1])]
        else:
            data = Ytrain
        np.savetxt(path + file_name + '.csv', data , header=''.join(header), delimiter=',', comments='')


    @staticmethod
    def export_track(name, track):
        path = Experiments.get_output_path() + name + '/'
        check_dir_exists(path)
        file_name = 'obj_track_'
        np.savetxt(path + file_name + '.csv', np.array([track]).T,
                   header='objective'
                   , delimiter=',', comments='')

    @staticmethod
    def export_model(model, name):
        path = Experiments.get_output_path() + name + '/'
        check_dir_exists(path)
        file_name = 'model_'
        if model is not None:
            with open(path + file_name + '.csv', 'w') as fp:
                f = csv.writer(fp, delimiter=',')
                f.writerow(['#model', model.__class__])
                params = model.get_all_params()
                param_names = model.get_all_param_names()
                for j in range(len(params)):
                    f.writerow([param_names[j], params[j]])


    @staticmethod
    def export_test(name, X, Ytrue, Ypred, Yvar_pred, nlpd, pred_names, export_X=False):
        path = Experiments.get_output_path() + name + '/'
        check_dir_exists(path)
        file_name = 'test_'
        out = []
        out.append(Ytrue)
        out += Ypred
        out += Yvar_pred
        out += [nlpd]
        header =  ['Ytrue%d,' % (j) for j in range(Ytrue.shape[1])] + \
            ['Ypred_%s_%d,' % (m, j) for m in pred_names for j in range(Ypred[0].shape[1])] + \
            ['Yvar_pred_%s_%d,' % (m, j) for m in pred_names for j in range(Yvar_pred[0].shape[1])] + \
            ['nlpd,']


        if export_X:
            out.append(X)
            header += ['X%d,' % (j) for j in range(X.shape[1])]

        header = ''.join(header)
        out = np.hstack(out)
        np.savetxt(path + file_name + '.csv', out
                   , header=header
                   , delimiter=',', comments='')


    @staticmethod
    def export_configuration(name, config):
        path = Experiments.get_output_path() + name + '/'
        check_dir_exists(path)
        file_name = path + 'config_' + '.csv'
        with open(file_name, 'wb') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=config.keys())
            writer.writeheader()
            writer.writerow(config)

    @staticmethod
    def get_ID():
        return id_generator(size=6)


    @staticmethod
    def opt_callback(name):
        def callback(model, current_iter, total_evals, delta_m, delta_s, obj_track):
            path = Experiments.get_output_path() + name + '/'
            check_dir_exists(path)
            pickle.dump(model.image(), open(path + 'model.dump', 'w'))
            pickle.dump({
                'current_iter': current_iter,
                'total_evals': total_evals,
                'delta_m': delta_m,
                'delta_s': delta_s,
                'obj_track': obj_track,
                'obj_fun': model.objective_function()
            },
                        open(path + 'opt.dump', 'w'))
        return callback


    @staticmethod
    def run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, run_id, num_inducing, num_samples,
                  sparsify_factor, to_optimize, trans_class, random_Z, logging_level, export_X,
                  latent_noise=0.001, opt_per_iter=None, max_iter=200, n_threads=1, model_image_file=None,
                  xtol=1e-3, ftol=1e-5, partition_size=3000, save_model = True, log_message = '',
                  bin_prior = 0.001, init_bin_s = 1.0):

        if opt_per_iter is None:
            opt_per_iter = {'mog': 40, 'hyp': 40, 'll': 40}
        folder_name = name + '_' + Experiments.get_ID()
        logger = Experiments.get_logger(folder_name, logging_level)

        logger.info(log_message)

        transformer = trans_class.get_transformation(Ytrain, Xtrain)
        Ytrain = transformer.transform_Y(Ytrain)
        Ytest = transformer.transform_Y(Ytest)
        Xtrain = transformer.transform_X(Xtrain)
        Xtest = transformer.transform_X(Xtest)

        Experiments.export_train(folder_name, transformer.untransform_X(Xtrain), transformer.untransform_Y(Ytrain), export_X)

        opt_max_fun_evals = None
        total_time = None
        timer_per_iter = None
        tracker = None
        export_model = False
        git_hash, git_branch = get_git()

        properties = {'method': method,
                   'sparsify_factor': sparsify_factor,
                   'sample_num': num_samples,
                   'll': cond_ll.__class__.__name__,
                   'opt_max_evals': opt_max_fun_evals,
                   'opt_per_iter': opt_per_iter,
                   'xtol': xtol,
                   'ftol': ftol,
                   'run_id': run_id,
                   'experiment': name,
                   'max_iter': max_iter,
                   'git_hash': git_hash,
                   'git_branch': git_branch,
                   'random_Z': random_Z,
                   'latent_noise:': latent_noise,
                   'model_init': model_image_file,
                   'bin_prior': bin_prior,
                   'bin_init_s': init_bin_s
                   }
        logger.info('experiment started for:' + str(properties))

        model_image = None
        current_iter = None
        if model_image_file is not None:
            model_image = pickle.load(open(model_image_file + 'model.dump'))
            opt_params = pickle.load(open(model_image_file + 'opt.dump'))
            current_iter = opt_params['current_iter']

        if save_model:
            callback = Experiments.opt_callback(folder_name)
        else:
            callback = None

        if model_image:
            logger.info('loaded model - iteration started from: ' + str(opt_params['current_iter']) +
                        ' Obj fun: ' + str(opt_params['obj_fun']) + ' fun evals: ' + str(opt_params['total_evals']))
        if method == 'full':
            m = SAVIGP_SingleComponent(Xtrain, Ytrain, num_inducing, cond_ll,
                                       kernel, num_samples, None, latent_noise, False, random_Z, n_threads=n_threads, image=model_image, partition_size=partition_size, logger = logger)
            _, timer_per_iter, total_time, tracker, total_evals = \
                Optimizer.optimize_model(m, opt_max_fun_evals, logger, to_optimize, xtol, opt_per_iter, max_iter, ftol, Experiments.opt_callback(folder_name), current_iter)
        if method == 'structured':
            m = StructureGP(Xtrain, Ytrain, num_inducing, cond_ll,
                                       kernel, num_samples, None, latent_noise, False, random_Z, n_threads=n_threads,
                                       image=model_image, partition_size=partition_size, logger = logger, bin_prior= bin_prior,
                                       init_bin_s=init_bin_s)
            _, timer_per_iter, total_time, tracker, total_evals = \
                Optimizer.optimize_model(m, opt_max_fun_evals, logger, to_optimize, xtol, opt_per_iter, max_iter, ftol, callback, current_iter)

        if method == 'mix1':
            m = SAVIGP_Diag(Xtrain, Ytrain, num_inducing, 1, cond_ll,
                            kernel, num_samples, None, latent_noise, False, random_Z, n_threads=n_threads, image=model_image, partition_size=partition_size)
            _, timer_per_iter, total_time, tracker, total_evals = \
                Optimizer.optimize_model(m, opt_max_fun_evals, logger, to_optimize, xtol, opt_per_iter, max_iter, ftol, Experiments.opt_callback(folder_name), current_iter)
        if method == 'mix2':
            m = SAVIGP_Diag(Xtrain, Ytrain, num_inducing, 2, cond_ll,
                            kernel, num_samples, None, latent_noise, False, random_Z, n_threads=n_threads, image=model_image, partition_size=partition_size)
            _, timer_per_iter, total_time, tracker, total_evals = \
                Optimizer.optimize_model(m, opt_max_fun_evals, logger, to_optimize, xtol, opt_per_iter, max_iter, ftol, Experiments.opt_callback(folder_name), current_iter)
        if method == 'gp':
            m = GPy.models.GPRegression(Xtrain, Ytrain, kernel[0])
            if 'll' in to_optimize and 'hyp' in to_optimize:
                m.optimize('bfgs')

        y_pred, var_pred, nlpd = m.predict(Xtest, Ytest)
        if not (tracker is None):
            Experiments.export_track(folder_name, tracker)
        Experiments.export_test(folder_name,
                                transformer.untransform_X(Xtest),
                                transformer.untransform_Y(Ytest),
                                [transformer.untransform_Y(y_pred)],
                                [transformer.untransform_Y_var(var_pred)],
                                transformer.untransform_NLPD(nlpd),
                                [''], export_X)

        if export_model and isinstance(m, SAVIGP):
            Experiments.export_model(m, folder_name)

        properties['total_time'] = total_time
        properties['time_per_iter'] = timer_per_iter
        properties['total_evals'] = total_evals
        Experiments.export_configuration(folder_name, properties)
        return folder_name, m

    @staticmethod
    def boston_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.boston_data()
        d = data[config['run_id'] - 1]
        names = []
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'boston'
        kernel = Experiments.get_kernels(Xtrain.shape[1], 1, True)
        # gaussian_sigma = np.var(Ytrain)/4 + 1e-4
        gaussian_sigma = 1.0
        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        cond_ll = UnivariateGaussian(np.array(gaussian_sigma))
        num_samples = 2000

        names.append(
            Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                  num_samples, sparsify_factor, ['hyp', 'mog', 'll'], MeanTransformation, True,
                                  config['log_level'], False, latent_noise=0.001,
                                  opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                  max_iter=200))
        return names

    @staticmethod
    def wisconsin_breast_cancer_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.wisconsin_breast_cancer_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'breast_cancer'

        # uncomment these lines to use softmax
        # kernel = Experiments.get_kernels(Xtrain.shape[1], 2, False)
        # Ytrain = np.array([(Ytrain[:,0] + 1) / 2, (-Ytrain[:,0] + 1) / 2]).T
        # Ytest = np.array([(Ytest[:,0] + 1) / 2, (-Ytest[:,0] + 1) / 2]).T
        # cond_ll = SoftmaxLL(2)

        # uncomment these lines to use logistic
        cond_ll = LogisticLL()
        kernel = Experiments.get_kernels(Xtrain.shape[1], 1, False)

        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000
        names.append(
            Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                  num_samples, sparsify_factor, ['mog', 'hyp'], IdentityTransformation, True,
                                  config['log_level'], False, latent_noise=0.001,
                                  opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                  max_iter=200))
        return names


    @staticmethod
    def mining_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.mining_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'mining'
        kernel = Experiments.get_kernels(Xtrain.shape[1], 1, False)

        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000
        cond_ll = LogGaussianCox(math.log(191./811))
        kernel[0].variance= 1.0
        kernel[0].lengthscale= 13516.

        names.append(
            Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                  num_samples, sparsify_factor, ['mog'], IdentityTransformation, True,
                                  config['log_level'], True, latent_noise=0.001,
                                  opt_per_iter={'mog': 15000, 'hyp': 25, 'll': 25},
                                  max_iter=1))
        return names


    @staticmethod
    def USPS_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.USPS_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'USPS'
        kernel = [ExtRBF(Xtrain.shape[1], variance=2, lengthscale=np.array((4.,)), ARD=False) for j in range(3)]
        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000
        cond_ll = SoftmaxLL(3)

        names.append(
            Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                  num_samples, sparsify_factor, ['mog', 'hyp'], IdentityTransformation, True,
                                  config['log_level'], False,  latent_noise=0.001,
                                  opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                  max_iter=300))


    @staticmethod
    def abalone_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.abalone_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'abalone'
        kernel = Experiments.get_kernels(Xtrain.shape[1], 1, False)

        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000

        cond_ll = WarpLL(np.array([-2.0485, 1.7991, 1.5814]),
                         np.array([2.7421, 0.9426, 1.7804]),
                         np.array([0.1856, 0.7024, -0.7421]),
                         np.log(0.1))

        names.append(
            Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                  num_samples, sparsify_factor, ['mog', 'hyp', 'll'], MinTransformation, True,
                                  config['log_level'], False, latent_noise=0.001,
                                  opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                  max_iter=200))


    @staticmethod
    def creep_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.creep_data()

        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'creep'
        scaler = preprocessing.StandardScaler().fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xtest = scaler.transform(Xtest)
        kernel = Experiments.get_kernels(Xtrain.shape[1], 1, True)

        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000

        cond_ll = WarpLL(np.array([3.8715, 3.8898, 2.8759]),
                         np.array([1.5925, -1.3360, -2.0289]),
                         np.array([0.7940, -4.1855, -3.0289]),
                         np.log(0.01))

        names.append(
            Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                  num_samples, sparsify_factor, ['mog', 'hyp', 'll'], MinTransformation, True,
                                  config['log_level'], False, latent_noise=0.001,
                                  opt_per_iter={'mog': 25, 'hyp': 25, 'll': 25},
                                  max_iter=200))


    @staticmethod
    def MNIST_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.mnist_data()
        names = []
        d = data[config['run_id'] - 1]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'mnist'

        # uncomment these lines to delete unused features
        # features_rm = np.array([])
        # for n in range(Xtrain.shape[1]):
        #     if Xtrain[:, n].sum() ==0:
        #         features_rm = np.append(features_rm, n)
        # Xtrain = np.delete(Xtrain, features_rm.astype(int), 1)
        # Xtest = np.delete(Xtest, features_rm.astype(int), 1)


        # uncomment these lines to change the resolution
        # res = 13
        # current_res = int(np.sqrt(Xtrain.shape[1]))
        # X_train_resized = np.empty((Xtrain.shape[0], res * res))
        # X_test_resized = np.empty((Xtest.shape[0], res * res))
        # for n in range(Xtrain.shape[0]):
        #     im = Image.fromarray(Xtrain[n, :].reshape((current_res, current_res)))
        #     im = im.resize((res, res))
        #     X_train_resized[n] = np.array(im).flatten()
        #
        # for n in range(Xtest.shape[0]):
        #     im = Image.fromarray(Xtest[n, :].reshape((current_res, current_res)))
        #     im = im.resize((res, res))
        #     X_test_resized[n] = np.array(im).flatten()
        #
        #
        # Xtrain = X_train_resized
        # Xtest = X_test_resized

        kernel = [ExtRBF(Xtrain.shape[1], variance=11, lengthscale=np.array((9.,)), ARD=False) for j in range(10)]
        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000
        cond_ll = SoftmaxLL(10)

        if 'n_thread' in config.keys():
            n_threads = config['n_thread']
        else:
            n_threads = 1

        if 'partition_size' in config.keys():
            partition_size = config['partition_size']
        else:
            partition_size = 3000


        image = None
        if 'image' in config.keys():
            image = config['image']

        names.append(
            Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                  num_samples, sparsify_factor, ['mog', 'hyp'], IdentityTransformation, False,
                                  config['log_level'], False,  latent_noise=0.001,
                                  opt_per_iter={'mog': 50, 'hyp': 10},
                                  max_iter=300, n_threads=n_threads, ftol=10,
                                  model_image_file=image, partition_size=partition_size))


    @staticmethod
    def struct_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)

        (ll_train,
        posterior_marginals_test,
        compute_error_nlm,
        ll_test,
        average_marginals,
        write_marginals,
        read_marginals,
        n_labels,
        Xtrain,
        Xtest,
         train_dataset,
         test_dataset)  = gpstruct_wrapper()
        name = 'struct'
        n_latent_processes = n_labels + 1

        Xtrain = np.array(Xtrain.todense())
        Xtest = np.array(Xtest.todense())
        num_latent_proc = n_labels
        # kernel = [ExtRBF(Xtrain.shape[1], variance=1.0, lengthscale=np.array(100), ARD=False) for j in range(num_latent_proc)]
        # kernel = [ExtRBF(Xtrain.shape[1], lengthscale = 0.0005, ARD=False, variance=1) for j in range(num_latent_proc)]

        kernel_variance = 0.001
        bin_prior = 0.001
        ini_bin_s = 1.0

        kernel = [Linear(Xtrain.shape[1], ARD=False, variances= kernel_variance) for j in range(num_latent_proc)]
        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 10000
        cond_ll = StructLL(ll_train, train_dataset, test_dataset)
        names = []
        image = None
        if 'image' in config.keys():
            image = config['image']


        Ytest_labels = np.hstack(np.array(test_dataset.Y))
        Ytest = np.zeros((Xtest.shape[0], test_dataset.n_labels))
        Ytest[np.arange(Xtest.shape[0]), Ytest_labels] = 1

        Ytrain_labels = np.hstack(np.array(train_dataset.Y))
        Ytrain = np.zeros((Xtrain.shape[0], train_dataset.n_labels))
        Ytrain[np.arange(Xtrain.shape[0]), Ytrain_labels] = 1
        names.append(
            Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, 1, num_inducing,
                                  num_samples, sparsify_factor, ['mog'], IdentityTransformation, True,
                                  config['log_level'], False,  latent_noise=0.001,
                                  opt_per_iter={'mog': 60, 'bin': 15},
                                  max_iter=1, n_threads=1,
                                  model_image_file=image,
                                  save_model=True,
                                  log_message=(str(kernel_variance)),
                                  bin_prior= bin_prior,
                                  init_bin_s=ini_bin_s

                                  )

        )

    @staticmethod
    def sarcos_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.sarcos_data()

        names = []
        d = data[0]
        Xtrain = d['train_X']
        Ytrain = d['train_Y']
        Xtest = d['test_X']
        Ytest = d['test_Y']
        name = 'sarcos'
        kernel = Experiments.get_kernels(Xtrain.shape[1], 3, False)

        # number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = 2000

        cond_ll =CogLL(0.1, 2, 1)

        if 'n_thread' in config.keys():
            n_threads = config['n_thread']
        else:
            n_threads = 1

        if 'partition_size' in config.keys():
            partition_size = config['partition_size']
        else:
            partition_size = 3000

        names.append(
            Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                  num_samples, sparsify_factor, ['mog', 'll', 'hyp'], MinTransformation, True,
                                  config['log_level'], False, latent_noise=0.001,
                                  opt_per_iter={'mog': 50, 'hyp': 10, 'll': 10},
                                  max_iter=200,
                                  partition_size=partition_size,
                                  n_threads=n_threads))


    @staticmethod
    def get_kernels(input_dim, num_latent_proc, ARD):
        return [ExtRBF(input_dim, variance=1, lengthscale=np.array((1.,)), ARD=ARD) for j in range(num_latent_proc)]

    @staticmethod
    def get_train_test(X, Y, n_train):
        data = np.hstack((X, Y))
        np.random.shuffle(data)
        Xn = data[:, :X.shape[1]]
        Yn = data[:, X.shape[1]:]
        return Xn[:n_train], Yn[:n_train], Xn[n_train:], Yn[n_train:]

