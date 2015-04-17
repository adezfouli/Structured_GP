from data_transformation import MeanTransformation, IdentityTransformation
from plot_results import PlotOutput
from savigp import SAVIGP
from savigp_diag import SAVIGP_Diag
from savigp_single_comp import SAVIGP_SingleComponent

__author__ = 'AT'

import csv
import GPy
from sklearn import preprocessing
from likelihood import MultivariateGaussian, UnivariateGaussian, LogisticLL, SoftmaxLL
from data_source import DataSource
import numpy as np
from optimizer import Optimizer
from plot import plot_fit
from savigp_prediction import SAVIGP_Prediction
from matplotlib.pyplot import show
from util import id_generator, check_dir_exists




class Experiments:

    @staticmethod
    def get_output_path():
        return '../../results/'

    @staticmethod
    def get_number_samples():
        return 2000

    @staticmethod
    def export_train(name, Xtrain, Ytrain):
        path = Experiments.get_output_path() + name +'/'
        check_dir_exists(path)
        file_name = 'train_'
        np.savetxt(path + file_name + '.csv', np.hstack((Ytrain, Xtrain))
                   , header=''.join(
                                    ['Y%d,'%(j) for j in range(Ytrain.shape[1])]+
                                    ['X%d,'%(j) for j in range(Xtrain.shape[1])]
                                    )
                    , delimiter=',', comments='')


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
        path = Experiments.get_output_path() + name +'/'
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
    def export_test(name, X, Ytrue, Ypred, Yvar_pred, pred_names):
        path = Experiments.get_output_path() + name +'/'
        check_dir_exists(path)
        file_name = 'test_'
        out = []
        out.append(Ytrue)
        out += Ypred
        out += Yvar_pred
        out.append(X)
        out = np.hstack(out)
        np.savetxt(path + file_name + '.csv', out
                   , header=''.join(
                                    ['Ytrue%d,'%(j) for j in range(Ytrue.shape[1])] +
                                    ['Ypred_%s_%d,'%(m, j) for m in pred_names for j in range(Ypred[0].shape[1])] +
                                    ['Yvar_pred_%s_%d,'%(m, j) for m in pred_names for j in range(Yvar_pred[0].shape[1])] +
                                    ['X%d,'%(j) for j in range(X.shape[1])]
                                    )
                    , delimiter=',', comments='')


    @staticmethod
    def export_configuration(name, config):
        path = Experiments.get_output_path() + name +'/'
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
    def run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, run_id, num_inducing, num_samples,
                  sparsify_factor, to_optimize, trans_class):

        transformer = trans_class.get_transformation(Ytrain, Xtrain)
        Ytrain = transformer.transform_Y(Ytrain)
        Ytest = transformer.transform_Y(Ytest)
        Xtrain = transformer.transform_X(Xtrain)
        Xtest = transformer.transform_X(Xtest)

        opt_max_fun_evals = None
        opt_iter = 200
        tol=1e-3
        total_time = None
        timer_per_iter = None
        verbose=False
        tracker=None
        if method == 'full':
            m = SAVIGP_SingleComponent(Xtrain, Ytrain, num_inducing, cond_ll,
                                         kernel, num_samples, None, 0.001, False)
            _, timer_per_iter, total_time, tracker = \
                Optimizer.optimize_model(m, opt_max_fun_evals, verbose, to_optimize, tol, opt_iter)
        if method == 'mix1':
            m = SAVIGP_Diag(Xtrain, Ytrain, num_inducing, 1, cond_ll,
                             kernel, num_samples, None, 0.001, False)
            _, timer_per_iter, total_time, tracker = \
                Optimizer.optimize_model(m, opt_max_fun_evals, verbose, to_optimize, tol, opt_iter)
        if method == 'mix2':
            m = SAVIGP_Diag(Xtrain, Ytrain, num_inducing, 2, cond_ll,
                             kernel, num_samples, None, 0.001, False)
            _, timer_per_iter, total_time, tracker = \
                Optimizer.optimize_model(m, opt_max_fun_evals, verbose, to_optimize, tol, opt_iter)
        if method == 'gp':
            m = GPy.models.GPRegression(Xtrain, Ytrain)
            if 'll' in to_optimize and 'hyp' in to_optimize:
                m.optimize('bfgs')

        y_pred, var_pred = m.predict(Xtest)
        folder_name =  name + '_' + Experiments.get_ID()
        if not (tracker is None):
            Experiments.export_track(folder_name, tracker)
        Experiments.export_train(folder_name, transformer.untransform_X(Xtrain), transformer.untransform_Y(Ytrain))
        Experiments.export_test(folder_name,
                                transformer.untransform_X(Xtest),
                                transformer.untransform_Y(Ytest),
                                [transformer.untransform_Y(y_pred)],
                                [transformer.untransform_Y_var(var_pred)], [''])
        if isinstance(m, SAVIGP):
            Experiments.export_model(m, folder_name)
        Experiments.export_configuration(folder_name, {'method': method,
                                                'sparsify_factor': sparsify_factor,
                                                'sample_num': num_samples,
                                                'll': cond_ll.__class__.__name__,
                                                'opt_max_evals': opt_max_fun_evals,
                                                'opt_iter': opt_iter,
                                                'tol': tol,
                                                'run_id': run_id,
                                                'experiment': name,
                                                'total_time': total_time,
                                                'time_per_iter': timer_per_iter
                                                },

                                        )
        return folder_name

    @staticmethod
    def boston_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.boston_data()
        names = []
        for d in data:
            Xtrain = d['train_X']
            Ytrain = d['train_Y']
            Xtest = d['test_X']
            Ytest = d['test_Y']
            name = 'boston'
            kernel = Experiments.get_kernels(Xtrain.shape[1], 1)
            gaussian_sigma = np.var(Ytrain)/4 + 1e-4

            #number of inducing points
            num_inducing = int(Xtrain.shape[0] * sparsify_factor)
            num_samples = Experiments.get_number_samples()
            cond_ll = UnivariateGaussian(np.array(gaussian_sigma))

            names.append(Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                         num_samples, sparsify_factor, ['mog', 'hyp', 'll'], MeanTransformation))
        return names

    @staticmethod
    def wisconsin_breast_cancer_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        data = DataSource.wisconsin_breast_cancer_data()
        names = []
        for d in data:
            Xtrain = d['train_X']
            Ytrain = d['train_Y']
            Xtest = d['test_X']
            Ytest = d['test_Y']
            name = 'breast_cancer'
            kernel = Experiments.get_kernels(Xtrain.shape[1], 1)

            #number of inducing points
            num_inducing = int(Xtrain.shape[0] * sparsify_factor)
            num_samples = Experiments.get_number_samples()
            cond_ll = LogisticLL()

            names.append(Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, d['id'], num_inducing,
                                     num_samples, sparsify_factor, ['mog', 'hyp'], IdentityTransformation))
        return names


    @staticmethod
    def USPS_data(config):
        method = config['method']
        sparsify_factor = config['sparse_factor']
        np.random.seed(12000)
        Xtrain, Ytrain, Xtest, Ytest = DataSource.USPS_data()
        Xtrain = preprocessing.scale(Xtrain)
        Xtest = preprocessing.scale(Xtest)

        name = 'USPS_' + Experiments.get_ID()
        kernel = Experiments.get_kernels(Xtrain.shape[1], 3)

        #number of inducing points
        num_inducing = int(Xtrain.shape[0] * sparsify_factor)
        num_samples = Experiments.get_number_samples()
        cond_ll = SoftmaxLL()

        return Experiments.run_model(Xtest, Xtrain, Ytest, Ytrain, cond_ll, kernel, method, name, num_inducing,
                                     num_samples, sparsify_factor, ['mog', 'hyp'])


    @staticmethod
    def get_kernels(input_dim, num_latent_proc):
        return [GPy.kern.RBF(input_dim, variance=1, lengthscale=np.array((1.,))) for j in range(num_latent_proc)]

    @staticmethod
    def gaussian_1D_data():
        gaussian_sigma = 0.2
        np.random.seed(12000)
        X, Y = DataSource.normal_1D_data(1000, gaussian_sigma)
        X = preprocessing.scale(X)
        Y = preprocessing.scale(Y)
        Xtrain, Ytrain, Xtest, YTest = Experiments.get_train_test(X, Y, 300)
        kernel = [GPy.kern.RBF(1, variance=0.5, lengthscale=np.array((0.2,)))]
        m = SAVIGP_SingleComponent(Xtrain, Ytrain, Xtrain.shape[0], MultivariateGaussian(np.array([[gaussian_sigma]])),
                                 kernel, 10000, None)
        Optimizer.optimize_model(m, 10000, True, ['mog', 'hyp'])
        plot_fit(m)
        show(block=True)

    @staticmethod
    def gaussian_1D_data_diag():
        sigma = 0.2
        np.random.seed(12000)
        X, Y = DataSource.normal_1D_data(20, sigma)
        X = preprocessing.scale(X)
        Y = preprocessing.scale(Y)
        Xtrain, Ytrain, Xtest, YTest = Experiments.get_train_test(X, Y, 20)
        kernel = [GPy.kern.RBF(1, variance=0.5, lengthscale=np.array((0.2,)))]
        m = SAVIGP_Diag(Xtrain, Ytrain, Xtrain.shape[0], 1, MultivariateGaussian(np.array([[sigma]])),
                                 kernel, 10000, None)
        Optimizer.optimize_model(m, 10000, True, ['mog', 'hyp', 'll'])
        plot_fit(m)
        gp = SAVIGP_Prediction.gpy_prediction(X, Y, sigma, kernel[0])
        gp.plot()
        show(block=True)


    @staticmethod
    def get_train_test(X, Y, n_train):
        data = np.hstack((X, Y))
        np.random.shuffle(data)
        Xn = data[:,:X.shape[1]]
        Yn = data[:,X.shape[1]:]
        return Xn[:n_train], Yn[:n_train], Xn[n_train:], Yn[n_train:]

