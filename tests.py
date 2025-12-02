import numpy as np
import importlib
import time
import keras
from keras import backend as K
import callbacks
import resnet_model
import vae_model
import predictor_model
import hybrid_model
import data_manager_cmems
import data_generator_cmems

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(resnet_model)
importlib.reload(predictor_model)
importlib.reload(vae_model)
importlib.reload(callbacks)


def test_data_generator():
    print('testing data generator')
    dmgr_cmems = \
        data_manager_cmems.DataManagerCMEMS(
            experiment_id='test',
            testing=True,
            force_rebuild=False,
        )
    dmgr_cmems.create_training_data()

    dgen_cmems = data_generator_cmems.DataGeneratorCMEMS(
        dm=dmgr_cmems,
        batch_size=4,
        lookback=4,
        mode='train',
        shuffle=True,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=10,
    )

    tic = time.time()
    print('random getitem calls')
    num_calls = 20
    pb_i = keras.utils.Progbar(num_calls, interval=0.1)
    for i in range(num_calls):
        pb_i.add(1)
        idx = np.random.randint(dgen_cmems.__len__())
        bx, by = dgen_cmems.__getitem__(idx)
    toc = time.time()
    elapsed = toc - tic
    print(elapsed)
    assert elapsed < 5


def test_resnet():
    K.clear_session()

    dmgr_cmems = \
        data_manager_cmems.DataManagerCMEMS(
            experiment_id='test/resnet',
            testing=True,
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
    resnet.summary()
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
        epochs=1,
        validation_data=dgen_test,
        callbacks=[
            analysis_callback,
            model_checkpoint_callback,
        ]
    )
    analysis_callback.plot_history(hist)


def test_vae():
    K.clear_session()

    dmgr_cmems = \
        data_manager_cmems.DataManagerCMEMS(
            experiment_id='test/vae',
            testing=True,
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

    vae = vae_model.VAE(data_gen=dgen_train)

    vae.build_model("betaVAE")
    vae.summary()
    vae.compile(vae.compiler)

    analysis_callback = callbacks.AnalysisVAE(data_gen=dgen_test,
                                              plot=[
                                                  'reconstruction',
                                                  'spectra',
                                              ]
                                              )

    checkpoint_filepath = \
        f'{dmgr_cmems.dirs["checkpoints"]}/checkpoint.vae.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    hist = vae.fit(
        x=dgen_train,
        epochs=1,
        validation_data=dgen_test,
        callbacks=[
            analysis_callback,
            model_checkpoint_callback,
        ]
    )

    analysis_callback.plot_history(hist)


def test_predictor():
    K.clear_session()

    dmgr_cmems = \
        data_manager_cmems.DataManagerCMEMS(
            experiment_id='test/predictor',
            testing=True,
        )

    dmgr_cmems.create_training_data()
    dgen_args = {
        'dm': dmgr_cmems,
        'batch_size': 4,
        'lookback': 4,
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
        'experiments/test/vae/checkpoints/checkpoint.vae.keras'
    # vae.load_weights(checkpoint_filepath)

    predictor = predictor_model.Predictor(data_gen=dgen_train,
                                          vae_model=vae)

    predictor.build_model("predictor")
    predictor.summary()
    predictor.compile(predictor.compiler)

    analysis_callback = callbacks.AnalysisPredictor(data_gen=dgen_test,
                                                    plot=[
                                                        # 'reconstruction',
                                                        # 'spectra',
                                                    ]
                                                    )

    checkpoint_filepath = \
        f'{dmgr_cmems.dirs["checkpoints"]}/checkpoint.predictor.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    predictor.fit(
        x=dgen_train,
        epochs=1,
        validation_data=dgen_test,
        callbacks=[
            analysis_callback,
            model_checkpoint_callback,
        ]
    )


def test_hybrid():
    K.clear_session()

    dmgr_cmems = \
        data_manager_cmems.DataManagerCMEMS(
            experiment_id='test/predictor',
            testing=True,
        )

    dmgr_cmems.create_training_data()
    dgen_args = {
        'dm': dmgr_cmems,
        'batch_size': 4,
        'lookback': 4,
        'shuffle': True,
        'use_multiprocessing': True,
        'workers': 4,
        'max_queue_size': 10,
    }

    dgen_train, dgen_test = \
        data_generator_cmems.getter(**dgen_args)

    resnet = resnet_model.ResNet(data_gen=dgen_train)
    resnet.build_model("ResNet")

    vae = vae_model.VAE(data_gen=dgen_train)
    vae.build_model("betaVAE")

    predictor = predictor_model.Predictor(data_gen=dgen_train,
                                          vae_model=vae)
    predictor.build_model("predictor")

    hybrid = hybrid_model.Hybrid(data_gen=dgen_train,
                                 resnet_model=resnet,
                                 predictor_model=predictor)

    hybrid.build_model("hybrid")
    hybrid.compile(hybrid.compiler)

    dmd_callback = callbacks.DMD(data_gen=dgen_train)

    hybrid.fit(
        x=dgen_train,
        epochs=1,
        validation_data=dgen_test,
        callbacks=[
            dmd_callback,
        ]
    )


# test_vae()
# test_predictor()
# test_data_generator()
# test_resnet()
# test_hybrid()
