import os
os.system('export MKL_NUM_THREADS=1')
os.system('export OMP_NUM_THREADS=1')
import sys
from datetime import datetime
import time
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt

import keras

import data_manager as dm
reload(dm)

from ESN.ESN import ESN

# base experiment
base_exp = '20240718_164427_tf_multiply'
models_dir = f'experiments/{base_exp}/models'
ae_esn_dir = f'experiments/{base_exp}/ae_esn_experiments'
os.system(f'mkdir -p {ae_esn_dir}')

model_path_encoder = f'{models_dir}/encoder_res.keras'
model_path_decoder = f'{models_dir}/decoder_res.keras'
# load trained encoder and decoder
encoder = keras.models.load_model(model_path_encoder)
decoder = keras.models.load_model(model_path_decoder)

# get training data and metadata
data, params, scalers  = dm.create_training_data()

print('encode train and test data')
xHR_train = encoder.predict(data['train']['HR'])
xLR_train = encoder.predict(data['train']['LR'])
xHR_test = encoder.predict(data['test']['HR'])
xLR_test = encoder.predict(data['test']['LR'])
T_test, enclat, enclon, filters = xLR_test.shape


# !! order here is actually a hyperparameter, 'C' make most sense as
# !! it clusters spatial information from the different channels
T_train = len(data['train']['time'])
T_test = len(data['test']['time'])
xHR_train = xHR_train.reshape(T_train, -1, order='C')
xLR_train = xLR_train.reshape(T_train, -1, order='C')
xHR_test = xHR_test.reshape(T_test, -1, order='C')
xLR_test = xLR_test.reshape(T_test, -1, order='C')
N_feats_orig = xHR_train.shape[1]

# Remove zero columns
nonzero_ids = np.where(np.sum(xHR_train, axis=0)!=0)[0]
xHR_train = xHR_train[:,nonzero_ids]
xLR_train = xLR_train[:,nonzero_ids]
xLR_test = xLR_test[:,nonzero_ids]
N_feats = xHR_train.shape[1]


# !! TODO factorize this somewhere
model_type = 'ESNc'


history = 8000

if (model_type == 'DMDc' or
    model_type == 'ESNc'):
    # !! another hyperparameter
    control_amp = 1
    trainU = np.hstack((xHR_train[-history:-1,] ,
                        xLR_train[-history+1:,] * control_amp))

elif (model_type == 'DMD' or
      model_type == 'ESN'):
    trainU = xHR_train[-history:-1,]

elif model_type == 'corr_only':
    raise Exception('not implemented')
     # trainU = X_LR[train_range_p,:]

trainY = xHR_train[-history+1:,]

esn_pars = {}
if (model_type == 'DMD' or
    model_type == 'DMDc' or
    model_type == 'corr_only'):
    esn_pars['dmdMode'] = True
else:
    esn_pars['dmdMode'] = False

if (model_type == 'DMD' or
    model_type == 'DMDc' or
    model_type == 'corr_only' or
    model_type == 'ESNc'):
    esn_pars['feedThrough'] = True
else:
    esn_pars['feedThrough'] = False

if model_type == 'ESNc':
    esn_pars['ftRange'] = range(N_feats,
                                2*N_feats)

#!! TODO Essential hyperparameter!!
esn_pars['scalingType']     = 'minMax1'
esn_pars['Nr']              = 5000
esn_pars['rhoMax']          = 0.4
esn_pars['alpha']           = 0.2
esn_pars['entriesPerRow']   = 7
esn_pars['tikhonov_lambda'] = 1e-2
esn_pars['squaredStates']   = 'even'
esn_pars['inputMatrixType'] = 'balancedSparse'
esn_pars['fCutoff']         = 0.01

esn = ESN(esn_pars['Nr'], trainU.shape[1], trainY.shape[1])
esn.setPars(esn_pars)
esn.initialize()
esn.train(trainU, trainY)

# -------------------------------------------------------
# CREATE PREDICTIONS
# -------------------------------------------------------
predY = np.zeros((T_test, N_feats_orig))
esn_state = esn.X[-1,:].copy()
print(np.linalg.norm(esn_state))

# initialization:
xk = xHR_train[-1,]

verbosity = 400
for i in range(T_test):

    # from data:
    Pxk = xLR_test[i,]

    if (model_type == 'DMDc' or
        model_type == 'ESNc' ):
        u_in = np.append(xk.squeeze(),
                         Pxk.squeeze() * control_amp)
    elif (model_type == 'DMD' or
          model_type == 'ESN'):
        u_in = xk.squeeze()

    elif model_type == 'corr_only':
        raise Exception('not implemented')
        # u_in = Pyk.squeeze()

    u_in       = np.expand_dims(u_in, axis=0)
    u_in       = esn.scaleInput(u_in)
    esn_state  = esn.update(esn_state, u_in)
    u_out      = esn.apply(esn_state, u_in)
    u_out      = np.expand_dims(u_out, axis=0)
    yk         = esn.unscaleOutput(u_out)

    xk = yk
    predY[i,nonzero_ids] = yk
    if not i % verbosity:
        print(f'{i} / {T_test}')

Y=predY.reshape(T_test, enclat, enclon, filters, order='C')
inputs = [Y, data['test']['LR']]
print('decoding predictions')
D = decoder.predict(inputs)
X=xHR_test.reshape(T_test, enclat, enclon, filters, order='C')

plt.close('all')
figsize=(11,9)
fig = plt.figure(figsize=figsize)
plt.subplot(3,2,1)
tid = 300
chn = 3
t_mse = T_test
h=plt.imshow(X[tid,:,:,chn], cmap='binary')
plt.colorbar(h)
plt.gca().invert_yaxis()
plt.subplot(3,2,3)
h=plt.imshow(Y[tid,:,:,chn], cmap='binary')
plt.colorbar(h)
plt.gca().invert_yaxis()
plt.subplot(3,2,5)
# h=plt.imshow(X[tid,:,:,chn]-Y[tid,:,:,chn], cmap='binary')
# plt.colorbar(h)
# plt.gca().invert_yaxis()
MSE = np.sqrt(np.sum(np.square(X-Y),axis=(1,2,3)))
plt.plot(MSE[:t_mse])
plt.gca().set_ylim([0,8])
plt.grid()

plt.subplot(3,2,2)
Z = data['test']['HR']
h=plt.imshow(Z[tid,:,:,0], cmap='viridis')
plt.colorbar(h)
plt.gca().invert_yaxis()
plt.subplot(3,2,4)
h=plt.imshow(D[tid,:,:,0], cmap='viridis')
plt.colorbar(h)
plt.gca().invert_yaxis()
plt.subplot(3,2,6)
# h=plt.imshow(Z[tid,:,:,0]-D[tid,:,:,0], cmap='viridis')
# plt.colorbar(h)
# plt.gca().invert_yaxis()
MSE = np.sqrt(np.sum(np.square(Z-D),axis=(1,2,3)))
plt.plot(MSE[:t_mse])
plt.gca().set_ylim([0,7])

plt.grid()

plt.tight_layout()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
figname = f'{ae_esn_dir}/results_{timestamp}.png'
print(figname)
plt.savefig(figname)
plt.pause(1)

