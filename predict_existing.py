import autoencoder_original
from importlib import reload
reload(autoencoder_original)
from autoencoder_original import AE_Experiment

model = {'folder' : 'experiments/test-default/models',
         'postfix' : 'trial_26_20240917_110616'}

exp = AE_Experiment(existing_model=model,
                    exp_name='test_restart',
                    tuning_config='default',
                    detide=False,
                    compute_data=False)
exp.build_and_run_model(predict_only=True)
exp.create_movie() # broken
