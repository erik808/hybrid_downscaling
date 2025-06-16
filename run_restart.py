from importlib import reload
import ae_experiment
import pytest
reload(ae_experiment)
from ae_experiment import AE_Experiment


if __name__=="__main__":
    
    model = {
        'folder' : 'experiments/test_RNN-default/models/',
        'postfix' : 'trial_45_20241101_203528'
    }

    exp = AE_Experiment(
        existing_model=model,
        exp_name='restart_test_RNN',
        tuning_config='default',
        feedthrough_type='hybrid')

    # adjust hyperparameters:
    test_pars = {'history'    : 'all',
                 'future'     : 'all',
                 'unroll_dim' : 0,
                 'lookback'   : 2,
                 'epochs'     : 1,
                 'batch_size' : 4}

    exp.hyper_params.update(test_pars)
    exp.build_and_run_model(predict_only=True,
                            evaluate=False)
    exp.create_movie()

