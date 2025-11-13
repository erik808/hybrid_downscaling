import keras
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

if len(sys.argv) < 2:
    experiment_id = 'train_resnet'
else:
    experiment_id = sys.argv[1]

dmgr_cmems = \
    data_manager_cmems.DataManagerCMEMS(
        experiment_id=experiment_id,
        testing=False,
    )
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

resnet = resnet_model.ResNet(data_gen=dgen_train)

resnet.build_model("ResNet")
resnet.summary()
resnet.compile(resnet.compiler)
# resnet_checkpoint = 'models/resnet/checkpoint.resnet.keras'
# resnet.load_weights(resnet_checkpoint)

analysis_callback = callbacks.AnalysisResNet(data_gen=dgen_test,
                                             plot=[
                                                 'reconstruction',
                                                 'spectra',
                                                 # 'timestepping',
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
    epochs=20,
    validation_data=dgen_test,
    callbacks=[
        analysis_callback,
        model_checkpoint_callback,
    ]
)

analysis_callback.plot_history(hist)
