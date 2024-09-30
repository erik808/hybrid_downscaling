import autoencoder_original
from importlib import reload
reload(autoencoder_original)
from autoencoder_original import AE_Experiment

model = {'folder' : 'experiments/blurring_v1-default/models',
         'postfix' : 'trial_39_20240927_141147'}

exp = AE_Experiment(existing_model=model,
                    exp_name='restart_blurring_v1',
                    tuning_config='default',
                    detide=False,
                    compute_data=False,
                    sigma=[1,1.5,1.5])
exp.hyper_params['future']='all'
exp.build_and_run_model(predict_only=True)
exp.create_movie()
