import importlib
import data_manager_cmems
import data_generator_cmems
import vae_model
import callbacks
import sys

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(vae_model)
importlib.reload(callbacks)

if len(sys.argv) < 2:
    experiment_id = 'train_vae'
else:
    experiment_id = sys.argv[1]

dmgr_cmems = \
    data_manager_cmems.DataManagerCMEMS(experiment_id=experiment_id)
dmgr_cmems.create_training_data(force_rebuild=False)

dgen_args = {
    'dm': dmgr_cmems,
    'batch_size': 4,
    'lookback': 1,
    'shuffle': True,
    'use_multiprocessing': True,
    'workers': 4,
    'max_queue_size': 10,
}

dgen_train, dgen_test = \
    data_generator_cmems.getter(**dgen_args)

vae = vae_model.VAE(data_gen=dgen_train)

vae.build_model("betaVAE")
vae.summary()
vae.compile(vae.compiler)
# breakpoint()
analysis_callback = callbacks.AnalysisVAE(data_gen=dgen_test,
                                          plot=[
                                              'reconstruction',
                                              'spectra',
                                          ])
hist = vae.fit(
    x=dgen_train,
    epochs=20,
    validation_data=dgen_test,
    callbacks=[
        analysis_callback,
    ]
)

analysis_callback.plot_history(hist)
