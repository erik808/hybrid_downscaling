from importlib import reload
import autoencoder_original
reload(autoencoder_original)
from autoencoder_original import AE_Experiment


def test_short_run():

    exp = AE_Experiment(
        exp_name='test_short_run',
        tuning_config='default',
        detide=False,
        compute_data=False,
        coarsening_method='gaussian_filter',
        sigma=[1,1.5,1.5],
        feedthrough_type='hybrid')

    # adjust hyperparameters:
    test_pars = {'history' : 1000,
                 'future' : 100,
                 'epochs' : 1,
                 'batch_size' : 4}

    exp.hyper_params.update(test_pars)

    exp.build_and_run_model()


if __name__=="__main__":
    test_short_run()
