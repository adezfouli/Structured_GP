import logging
from experiments import Experiments
from plot_results import PlotOutput
from multiprocessing.pool import ThreadPool, Pool


class ExperimentRunner:

    @staticmethod
    def get_configs():
        configs = []
        expr_names = ExperimentRunner.get_experiments()
        methods = ['full']
        sparse_factor = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        run_ids = [1, 2, 3, 4, 5]
        for e in expr_names:
            for m in methods:
                for s in sparse_factor:
                    for run_id in run_ids:
                        configs.append({'method': m,
                                        'sparse_factor': s,
                                        'method_to_run': e,
                                        'run_id': run_id,
                                        'log_level': ExperimentRunner.get_log_level()
                                        })

        return configs

    @staticmethod
    def get_experiments():
        # return [Experiments.boston_data.__name__]
        # return [Experiments.wisconsin_breast_cancer_data.__name__]
        return [Experiments.USPS_data.__name__]

    @staticmethod
    def get_log_level():
        # return logging.DEBUG
        return logging.INFO


    @staticmethod
    def get_expr_names():
        return str(ExperimentRunner.get_experiments())[2:6]

    @staticmethod
    def boston_experiment():
        Experiments.boston_data({'method': 'full', 'sparse_factor': 0.8, 'run_id': 3, 'log_level': logging.DEBUG})

    @staticmethod
    def wisconsin_breast_experiment():
        Experiments.wisconsin_breast_cancer_data({'method': 'mix1', 'sparse_factor': 1.0, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def abalone_experiment():
        Experiments.abalone_data({'method': 'full', 'sparse_factor': 1.0, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def USPS_experiment():
        Experiments.USPS_data({'method': 'mix1', 'sparse_factor': 1.0, 'run_id': 3, 'log_level': logging.DEBUG})

    @staticmethod
    def mining_experiment():
        Experiments.mining_data({'method': 'mix1', 'sparse_factor': 1.0, 'run_id': 1, 'log_level': logging.DEBUG})

    @staticmethod
    def plot():
        # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
        #                            lambda x: 'experiment' in x.keys() and x['experiment']== 'breast_cancer', False)
        PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
                                   lambda x: x['method'] == 'full', False)
        #
        # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
        #                            lambda x: x['c'] == '1' and (x['m'] in ['mix2', 'mix1', 'full', 'gp']), False)
        #
        # PlotOutput.plot_output_all('boston', Experiments.get_output_path(),
        #                        lambda x: x['c'] == '1' and (x['m'] in ['mix2', 'mix1', 'full', 'gp']), False)

def run_config(config):
    try:
        logger.info('started config: ' + str(config))
        getattr(Experiments, config['method_to_run'])(config)
        logger.info('finished config: ' + str(config))
    except Exception as e:
        logger.exception(config)


def run_config_serial(config):
    for c in config:
        run_config(c)

if __name__ == '__main__':
    logger = Experiments.get_logger('general_' + Experiments.get_ID(), logging.DEBUG)
    # n_process = 2
    # p = Pool(n_process)
    # p.map(run_config, ExperimentRunner.get_configs())
    # run_config_serial(ExperimentRunner.get_configs())
    # ExperimentRunner.boston_experiment()
    # ExperimentRunner.wisconsin_breast_experiment()
    # ExperimentRunner.USPS_experiment()
    # ExperimentRunner.mining_experiment()
    ExperimentRunner.plot()
