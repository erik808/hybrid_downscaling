import keras
from keras import backend as K
import numpy as np
import importlib
import data_manager_cmems
import data_generator_cmems
import vae_model
import predictor_model
import callbacks
import sys
import tools

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(vae_model)
importlib.reload(predictor_model)
importlib.reload(callbacks)
importlib.reload(tools)

experiment_id, seed, member = tools.input_handling(sys.argv)

K.clear_session()
keras.utils.set_random_seed(seed)
np.random.seed(seed)

dmgr_cmems = \
    data_manager_cmems.DataManagerCMEMS(experiment_id=experiment_id,
                                        testing=False,
                                        force_rebuild=False,
                                        )

dmgr_cmems.create_training_data()

dgen_args = {
    'dm': dmgr_cmems,
    'batch_size': 4,
    'lookback': 2,
    'shuffle': True,
    'use_multiprocessing': True,
    'workers': 4,
    'max_queue_size': 10,
}

dgen_train, dgen_test = \
    data_generator_cmems.getter(**dgen_args)

vae = vae_model.VAE(data_gen=dgen_train)
vae.build_model("betaVAE")
vae.summary(line_length=80)

# load saved weights
vae_checkpoint = \
    (f'experiment/vae_l4f64-64_spatial_bilinear/{member}'
     '/checkpoints/checkpoint.vae.keras')
print(f'loading weights from {vae_checkpoint}')
vae.load_weights(vae_checkpoint)

predictor = predictor_model.Predictor(data_gen=dgen_train,
                                      vae_model=vae)

predictor.build_model("predictor")
predictor.summary(line_length=80)
predictor.compile(predictor.compiler)

analysis_callback = callbacks.AnalysisPredictor(data_gen=dgen_test,
                                                dump_results=True,
                                                dump_truth=False,
                                                run_when='epoch_begin',
                                                )


dmd_train = callbacks.DMD(data_gen=dgen_train)
dmd_test = callbacks.DMD(data_gen=dgen_test)

checkpoint_filepath = \
    f'{dmgr_cmems.dirs["checkpoints"]}/checkpoint.predictor.keras'

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

hist = predictor.fit(
    x=dgen_train,
    epochs=1,
    validation_data=dgen_test,
    callbacks=[
        dmd_train,
        # dmd_test,
        analysis_callback,
        # model_checkpoint_callback,
    ]
)

analysis_callback.plot_history(hist)
