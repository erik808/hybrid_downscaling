import keras
import importlib
import data_manager_cmems
import data_generator_cmems
import vae_model
import predictor_model
import callbacks
import sys
from keras import backend as K

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(vae_model)
importlib.reload(predictor_model)
importlib.reload(callbacks)

K.clear_session()
keras.utils.set_random_seed(123)

if len(sys.argv) < 2:
    experiment_id = 'predictor_ESNcN10e3R1A1T5_exception'
else:
    experiment_id = sys.argv[1]

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

# load existing weights
vae_checkpoint = \
    'models/vae/l4k3f64-128spatial/checkpoint.vae.keras'
vae.load_weights(vae_checkpoint)

predictor = predictor_model.Predictor(data_gen=dgen_train,
                                      vae_model=vae)

predictor.build_model("predictor")
predictor.summary(line_length=80)
predictor.compile(predictor.compiler)

analysis_callback = callbacks.AnalysisPredictor(data_gen=dgen_test,
                                                plot=[
                                                    'reconstruction',
                                                    'spectra',
                                                ]
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
