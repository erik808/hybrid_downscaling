import importlib
import data_manager_cmems
import data_generator_cmems
import resnet_model

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(resnet_model)

dmgr_cmems = data_manager_cmems.DataManagerCMEMS()
dmgr_cmems.create_training_data(force_rebuild=False)

dgen_train = data_generator_cmems.DataGeneratorCMEMS(
    dm=dmgr_cmems,
    batch_size=4,
    lookback=4,
    mode='train',
    shuffle=True,
    # use_multiprocessing=True,
    # workers=4,
    # max_queue_size=10,
)

dgen_test = data_generator_cmems.DataGeneratorCMEMS(
    dm=dmgr_cmems,
    batch_size=4,
    lookback=4,
    mode='test',
    shuffle=True,
    # use_multiprocessing=True,
    # workers=4,
    # max_queue_size=10,
)

dgen_args = {
    'dm': dmgr_cmems,
    'batch_size': 4,
    'lookback': 4,
    'shuffle': True,
    'use_multiprocessing': True,
    'workers': 16,
    'max_queue_size': 20,
}


def wrap_data_generators(**args):
    dgen_train = data_generator_cmems.DataGeneratorCMEMS(mode='train',
                                                         **args)
    dgen_test = data_generator_cmems.DataGeneratorCMEMS(mode='test',
                                                        **args)
    return dgen_train, dgen_test


dgen_train, dgen_test = wrap_data_generators(**dgen_args)

resnet = resnet_model.ResNet(dgen_train)
resnet.build_model()
resnet.summary()
resnet.compile(resnet.compiler)

hist = resnet.fit(
    x=dgen_train,
    epochs=1,
    validation_data=dgen_test,
)
