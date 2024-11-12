import autoencoder_original
from autoencoder_original import AE_Experiment



model = {'folder' : 'experiments/gaussian_FT_only_2lrs-default/models/',
         'postfix' : 'trial_0_20241005_232914'}

model = {'folder' : 'experiments/gaussian_FT_hybrid_2lrs-default/models/',
         'postfix' : 'trial_0_20241005_232903'}

model = {'folder' : 'experiments/gaussian_inactive_FT-default/models/',
         'postfix' : 'trial_1_20241005_204321'}

model = {
    'folder' : 'experiments/test_RNN-default/models/',
    'postfix' : 'trial_45_20241101_203528'
}


exp = AE_Experiment(existing_model=model,
                    exp_name='restart_RNN',
                    tuning_config='default',
                    detide=False,
                    compute_data=False,
                    coarsening_method='gaussian_filter',
                    truncation=100,
                    sigma=[1,1.5,1.5],
                    feedthrough_type='hybrid')

exp.hyper_params['future']='all'

exp.build_and_run_model(predict_only=True,
                        evaluate=False)
exp.create_movie()
