import importlib
import data_manager_cmems
import data_generator_cmems
import vae_model


importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(vae_model)

dmgr_cmems = data_manager_cmems.DataManagerCMEMS()
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
# vae.compile(vae.compiler)

# hist = vae.fit(
#     x=dgen_train,
#     epochs=2,
#     validation_data=dgen_test,
# )
