import keras
from keras import backend as K
import numpy as np
import importlib
import data_manager_cmems
import data_generator_cmems
import resnet_model
import callbacks
import sys

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(resnet_model)
importlib.reload(callbacks)

experiment_id = 'train_resnet'
seed = 123

if len(sys.argv) > 1:
    experiment_id = sys.argv[1]
if len(sys.argv) > 2:
    seed = sys.argv[2]

K.clear_session()
keras.utils.set_random_seed(seed)
np.random.seed(seed)

dmgr_cmems = \
    data_manager_cmems.DataManagerCMEMS(
        experiment_id=experiment_id,
        testing=False,
        force_rebuild=False,
    )
dmgr_cmems.create_training_data()

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

resnet = resnet_model.ResNet(data_gen=dgen_train)

resnet.build_model("ResNet")
resnet.summary(expand_nested=True)
resnet.compile(resnet.compiler)

analysis_callback = callbacks.AnalysisResNet(data_gen=dgen_test,
                                             plot=[
                                                 # 'reconstruction',
                                                 # 'spectra',
                                             ]
                                             )

checkpoint_filepath = \
    f'{dmgr_cmems.dirs["checkpoints"]}/checkpoint.resnet.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

hist = resnet.fit(
    x=dgen_train,
    epochs=50,
    shuffle=False,
    validation_data=dgen_test,
    callbacks=[
        analysis_callback,
        model_checkpoint_callback,
    ]
)

analysis_callback.plot_history(hist)
