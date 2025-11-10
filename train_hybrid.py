# import keras
import importlib
import data_manager_cmems
from keras import backend as K
import data_generator_cmems
import resnet_model
import vae_model
import predictor_model
import hybrid_model
import callbacks
import sys

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)
importlib.reload(resnet_model)
importlib.reload(vae_model)
importlib.reload(predictor_model)
importlib.reload(hybrid_model)
importlib.reload(callbacks)

K.clear_session()

if len(sys.argv) < 2:
    experiment_id = 'train_hybrid'
else:
    experiment_id = sys.argv[1]

dmgr_cmems = \
    data_manager_cmems.DataManagerCMEMS(
        experiment_id=experiment_id,
        testing=True)
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

resnet = resnet_model.ResNet(data_gen=dgen_train)
resnet.build_model("ResNet")
resnet.summary()

vae = vae_model.VAE(data_gen=dgen_train)
vae.build_model("betaVAE")
vae.summary()

# load existing weights
checkpoint_filepath = \
    'experiments/train_vae/checkpoints/checkpoint.vae.keras'
vae.load_weights(checkpoint_filepath)

predictor = predictor_model.Predictor(data_gen=dgen_train,
                                      vae_model=vae)
predictor.build_model("predictor")
predictor.summary()

# create hybrid
hybrid = hybrid_model.Hybrid(data_gen=dgen_train,
                             resnet_model=resnet,
                             predictor_model=predictor)

hybrid.build_model("hybrid")
hybrid.summary()
hybrid.compile(hybrid.compiler)
breakpoint()

hist = hybrid.fit(
    x=dgen_train,
    epochs=1,
    validation_data=dgen_test,
    # callbacks=[
    #     analysis_callback,
    #     model_checkpoint_callback,
    # ]
)
