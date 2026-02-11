import keras
from keras import backend as K
import numpy as np
import importlib
import data_manager_cmems
import data_generator_cmems
import resnet_model
import callbacks
import sys
import tools

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(resnet_model)
importlib.reload(callbacks)
importlib.reload(tools)

experiment_id, seed, member = tools.input_handling(sys.argv)

K.clear_session()
keras.utils.set_random_seed(seed)
np.random.seed(seed)

inference_only = True

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

checkpoint_filepath = \
    f'{dmgr_cmems.dirs["checkpoints"]}/checkpoint.resnet.keras'

resnet.build_model("ResNet")
if inference_only:
    print('loading weights')
    resnet.load_weights(checkpoint_filepath)

resnet.summary(expand_nested=True)
resnet.compile(resnet.compiler)

run_when = 'epoch_begin' if inference_only else 'epoch_end'

analysis_callback = callbacks.AnalysisResNet(data_gen=dgen_test,
                                             dump_results=True,
                                             dump_truth=False,
                                             run_when=run_when,
                                             )

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
