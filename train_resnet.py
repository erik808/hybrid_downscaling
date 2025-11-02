import importlib
import data_manager_cmems
import data_generator_cmems
import resnet_model
import callbacks

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(resnet_model)
importlib.reload(callbacks)

dmgr_cmems = data_manager_cmems.DataManagerCMEMS(experiment_id='train_resnet')
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

analysis_callback = callbacks.AnalysisResNet(data_gen=dgen_test)

hist = resnet.fit(
    x=dgen_train,
    epochs=10,
    validation_data=dgen_test,
    callbacks=[
        analysis_callback,
    ]
)
