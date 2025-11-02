import importlib
import data_manager_cmems
import data_generator_cmems
import vae_model
import callbacks

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(vae_model)
importlib.reload(callbacks)

dmgr_cmems = data_manager_cmems.DataManagerCMEMS(experiment_id='train_vae')
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

analysis_callback = callbacks.AnalysisVAE(data_gen=dgen_test)

hist = vae.fit(
    x=dgen_train,
    epochs=10,
    validation_data=dgen_test,
    callbacks=[
        analysis_callback,
    ]
)
