from importlib import reload
from datetime import datetime

import pytest
import keras
import glob
import os

import ae_experiment
reload(ae_experiment)
from ae_experiment import AE_Experiment

# set a seed
keras.utils.set_random_seed(123)

exp_name = 'test_everything'

# decorator to make sure functions clean on ending
def clean_on_end(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        cleanup()
    return wrapper


@clean_on_end
def test_short_run():
    exp = AE_Experiment(
        exp_name=exp_name,
        compute_data=False,
        coarsening_method='gaussian_filter',
        sigma=[1,1.5,1.5],
        feedthrough_type='hybrid',
        testing_mode=True)

    # adjust hyperparameters:
    test_pars = {
        'history' : 200,
        'future' : 100,
        'epochs' : 1,
        'unroll_dim' : 0,
        'lookback' : 1,
        'noise_stddev' : 0.0,
        'dropout_rate' : 0.0,
        'num_conv_blocks' : 1,
        'kernel_size' : (3,3),
        'downsample_stride' : (2,2),
        'conv_layers_per_block' : 1,
        'num_filters' : 32,
        'num_filters_last' : 112,
        'batch_size' : 4,
        'RNN_model' : 'RNN',
        'latent_space_dim' : 8,
        'num_feedthrough_filters' : 112,
        'num_feedthrough_layers' : 1,
        'num_output_layers' : 1,
        'l2_lambda' : 0.0,
        'learning_rate' : 0.002,
    }

    exp.hyper_params.update(test_pars)
    err = exp.build_and_run_model()
    assert err < 30

@clean_on_end
def test_save_load():

    exp = AE_Experiment(
        exp_name=exp_name,
        compute_data=False,
        coarsening_method='gaussian_filter',
        sigma=[1,1.5,1.5],
        feedthrough_type='hybrid',
        testing_mode=True)

    # adjust hyperparameters:
    test_pars = {
        'history' : 200,
        'future' : 100,
        'epochs' : 1,
        'unroll_dim' : 0,
        'lookback' : 1,
        'noise_stddev' : 0.0,
        'dropout_rate' : 0.0,
        'num_conv_blocks' : 1,
        'kernel_size' : (3,3),
        'downsample_stride' : (2,2),
        'conv_layers_per_block' : 1,
        'num_filters' : 32,
        'num_filters_last' : 112,
        'batch_size' : 4,
        'RNN_model' : 'RNN',
        'latent_space_dim' : 8,
        'num_feedthrough_filters' : 112,
        'num_feedthrough_layers' : 1,
        'num_output_layers' : 1,
        'l2_lambda' : 0.0,
        'learning_rate' : 0.002,
    }

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


def cleanup():
    print('cleanup:')

    today = datetime.now().strftime('%Y%m%d')

    files = []
    for ext in ['.dill', '.keras']:
        files.extend(
            glob.glob(os.path.join(f'experiments/{exp_name}', "**", f"*{ext}"),
                      recursive=True)
        )

    for dfile in files:
        if today in dfile:
            print(f'deleting {dfile}')
            os.remove(dfile)

if __name__=="__main__":
    test_short_run()
    test_save_load()
