import keras
import importlib
import data_manager_cmems
import data_generator_cmems
import vae_model
import rnn_model
import callbacks
import sys

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(vae_model)
importlib.reload(rnn_model)
importlib.reload(callbacks)

if len(sys.argv) < 2:
    experiment_id = 'train_rnn'
else:
    experiment_id = sys.argv[1]

dmgr_cmems = \
    data_manager_cmems.DataManagerCMEMS(experiment_id=experiment_id)
dmgr_cmems.create_training_data(force_rebuild=False)

dgen_args = {
    'dm': dmgr_cmems,
    'batch_size': 4,
    'lookback': 3,
    'shuffle': True,
    'use_multiprocessing': True,
    'workers': 4,
    'max_queue_size': 10,
}

dgen_train, dgen_test = \
    data_generator_cmems.getter(**dgen_args)

vae = vae_model.VAE(data_gen=dgen_train)
vae.build_model("betaVAE")

# load existing weights
checkpoint_filepath = \
    'experiments/train_vae/checkpoints/checkpoint.vae.keras'
vae.load_weights(checkpoint_filepath)

rnn = rnn_model.RNN(data_gen=dgen_train,
                    vae_model=vae)

rnn.build_model("RNN")
rnn.summary()
rnn.compile(rnn.compiler)

analysis_callback = callbacks.AnalysisRNN(data_gen=dgen_test,
                                          plot=[
                                              'reconstruction',
                                              'spectra',
                                          ]
                                          )

checkpoint_filepath = \
    f'{dmgr_cmems.dirs["checkpoints"]}/checkpoint.rnn.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

hist = rnn.fit(
    x=dgen_train,
    epochs=10,
    validation_data=dgen_test,
    callbacks=[
        analysis_callback,
        model_checkpoint_callback,
    ]
)

analysis_callback.plot_history(hist)
