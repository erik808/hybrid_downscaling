from importlib import reload

import autoencoder_original
import pytest
reload(autoencoder_original)
from autoencoder_original import AE_Experiment

# set a seed
import keras
keras.utils.set_random_seed(123)

def test_short_run():
    exp = AE_Experiment(
        exp_name='test_suite',
        tuning_config='default',
        detide=False,
        compute_data=False,
        coarsening_method='gaussian_filter',
        sigma=[1,1.5,1.5],
        feedthrough_type='hybrid',
        testing_mode=True)

    # adjust hyperparameters:
    test_pars = {'history' : 'all',
                 'future' : 100,
                 'epochs' : 2,
                 'unroll_dim' : 0,
                 'lookback' : 5,
                 'noise_stddev' : 0.0,
                 'dropout_rate' : 0.0,
                 'num_conv_blocks' : 2,
                 'conv_layers_per_block' : 1,
                 'num_filters' : 112,
                 'num_filters_last' : 16,
                 'batch_size' : 4,
                 'RNN_model' : 'RNN',
                 'RNN_dim' : 64,
                 'num_feedthrough_filters' : None,
                 'num_feedthrough_layers' : 2,
                 'L2_lambda' : 0.0,
                 'learning_rate' : 0.002,}

    exp.hyper_params.update(test_pars)
    err = exp.build_and_run_model()
    assert err < 30

def test_save_load():

    exp = AE_Experiment(
        exp_name='test_suite',
        tuning_config='default',
        detide=False,
        compute_data=False,
        coarsening_method='gaussian_filter',
        sigma=[1,1.5,1.5],
        feedthrough_type='hybrid',
        testing_mode=True)

    # adjust hyperparameters:
    test_pars = {'history' : 128,
                 'future' : 16,
                 'lookback' : 1,
                 'conv_layers_per_block' : 1,
                 'epochs' : 1,
                 'batch_size' : 4,
                 'RNN_reduction_factor' : 8}

    exp.hyper_params.update(test_pars)

    err = exp.build_and_run_model()
    assert err < 30

    modelpath = exp.save_path_autoencoder.split('/autoencoder_')
    modelfolder = modelpath[0]
    modelpostfix = modelpath[1].split('.keras')[0]

    model = {'folder' : modelfolder,
             'postfix' : modelpostfix}

    exp = AE_Experiment(
        existing_model=model,
        exp_name='test_save_load',
        tuning_config='default',
        detide=False,
        compute_data=False,
        coarsening_method='gaussian_filter',
        sigma=[1,1.5,1.5],
        feedthrough_type='hybrid',
        testing_mode=True)

    # adjust hyperparameters:
    test_pars = {'history' : 128,
                 'future' : 16,
                 'lookback' : 1,
                 'conv_layers_per_block' : 1,
                 'epochs' : 1,
                 'batch_size' : 4,
                 'RNN_reduction_factor' : 8}

    exp.hyper_params.update(test_pars)
    err = exp.build_and_run_model()
    assert err < 30

if __name__=="__main__":
    test_short_run()
    # test_save_load()
