import numpy as np
from ESN.ESN import ESN
import torch
import keras
from keras import layers
from keras import ops

hyperparams = { 'external' : {'model_type'      : 'ESN',
                              'training_length' : 25000,
                              'repetitions'     : 2,
                              'test_length'     : 4*24*10,
                              'reshape_order'   : 'C',
                              'decode_pred'     : True,
                              'bypass_mode'     : False,
                              'control_amp'     : 1 },

                'internal' : { 'Nr'                 : 5000,
                               'scalingType'        : 'none',
                               'rhoMax'             : 0.7,
                               'alpha'              : 0.3,
                               'avgDegree'          : 5,
                               'entriesPerRow'      : 50,
                               'noiseAmplitude'     : 0,
                               'tikhonov_lambda'    : 1,
                               'squaredStates'      : 'even',
                               'reservoirStateInit' : 'zero',
                               'inputMatrixType'    : 'balancedSparse',
                               'fCutoff'            : 0.01,
                               'Wconstruction'      : 'avgDegree'} }

class ESN_interface():

    def __init__(self, orig_data, enc_data, hyperparams,
                 encoder=None, decoder=None):

        _, self.enclat, self.enclon, self.filters = \
            enc_data['test']['LR'].shape

        self.orig_test_data = orig_data['test']['LR']

        self.encoder = encoder
        self.decoder = decoder

        self.reshape_order = hyperparams['external']['reshape_order']
        test_length = hyperparams['external']['test_length']

        self.T_train = len(orig_data['train']['time'])
        self.T_test = np.min([len(orig_data['test']['time']), test_length])

        # Reshape train and test data
        # !! reshape_order: 'C' make most sense as it clusters spatial
        # !! information from the different channels
        self.xHR_train = enc_data['train']['HR']\
            .reshape(self.T_train, -1, order=self.reshape_order)
        self.xLR_train = enc_data['train']['LR']\
            .reshape(self.T_train, -1, order=self.reshape_order)
        self.xHR_test = enc_data['test']['HR'][:self.T_test,]\
            .reshape(self.T_test, -1, order=self.reshape_order)
        self.xLR_test = enc_data['test']['LR'][:self.T_test,]\
            .reshape(self.T_test, -1, order=self.reshape_order)

        self.N_feats_orig = self.xHR_train.shape[1]

        # Remove zero columns
        self.nonzero_ids = np.where(np.sum(self.xHR_train, axis=0)!=0)[0]
        self.xHR_train = self.xHR_train[:,self.nonzero_ids]
        self.xLR_train = self.xLR_train[:,self.nonzero_ids]
        self.xLR_test = self.xLR_test[:,self.nonzero_ids]

        self.model_type = hyperparams['external']['model_type']

        self.history = hyperparams['external']['training_length']
        self.control_amp = hyperparams['external']['control_amp']

        if (self.model_type == 'DMDc' or
            self.model_type == 'ESNc'):
            # !! another hyperparameter
            self.trainU = np.hstack((self.xHR_train[-self.history:-1,] ,
                                     self.xLR_train[-self.history+1:,]
                                     * self.control_amp))

        elif (self.model_type == 'DMD' or
              self.model_type == 'ESN'):
            self.trainU = self.xHR_train[-self.history:-1,]

        elif self.model_type == 'corr_only':
            raise Exception('not implemented')
             # self.trainU = X_LR[train_range_p,:]

        self.trainY = self.xHR_train[-self.history+1:,]

        if (self.model_type == 'DMD' or
            self.model_type == 'DMDc' or
            self.model_type == 'corr_only'):
            hyperparams['internal']['dmdMode'] = True
        else:
            hyperparams['internal']['dmdMode'] = False

        if (self.model_type == 'DMD' or
            self.model_type == 'DMDc' or
            self.model_type == 'corr_only' or
            self.model_type == 'ESNc'):
            hyperparams['internal']['feedThrough'] = True
        else:
            hyperparams['internal']['feedThrough'] = False

        if self.model_type == 'ESNc':
            N_feats = self.xHR_train.shape[1]
            hyperparams['internal']['ftRange'] = range(N_feats,
                                                       2*N_feats)

        self.esn = ESN(hyperparams['internal']['Nr'],
                       self.trainU.shape[1],
                       self.trainY.shape[1])
        self.esn.setPars(hyperparams['internal'])
        self.esn.initialize()
        self.hyperparams = hyperparams
        # initialization is done

    def train(self):
        self.esn.train(self.trainU,
                       self.trainY)
        self.esn_state = self.esn.X[-1,:].copy()

    def create_predictions(self):

        predY = np.zeros((self.T_test, self.N_feats_orig))

        # esn state
        sk = self.esn.X[-1,:].copy()

        # initialization:
        xk = self.xHR_train[-1,]

        dec_pred = self.hyperparams['external']['decode_pred']

        verbosity = 400
        for i in range(self.T_test):
            # from data:
            Pxk = self.xLR_test[i,]
            if dec_pred:
                Pxk_dec = np.expand_dims(self.orig_test_data[i,], axis=0)
            else:
                Pxk_dec = None

            xk, sk, yk = self.step(xk, Pxk, sk, Pxk_dec)
            predY[i,self.nonzero_ids] = yk

            if not i % verbosity:
                print(f'{i} / {self.T_test}, decoding predictions: {dec_pred}')


        Y = predY.reshape(self.T_test, self.enclat, self.enclon,
                          self.filters, order=self.reshape_order)
        X = self.xHR_test.reshape(self.T_test, self.enclat, self.enclon,
                             self.filters, order=self.reshape_order)

        MSE = np.mean(np.sum(np.square(X-Y),axis=(1,2,3)))
        RMSE = np.sqrt(MSE)
        return Y, X, RMSE

    def step(self, xk, Pxk, sk, Pxk_dec=None):
        if (self.model_type == 'DMDc' or
            self.model_type == 'ESNc' ):
            u_in = np.append(xk.squeeze(),
                             Pxk.squeeze() * self.control_amp)
        elif (self.model_type == 'DMD' or
              self.model_type == 'ESN'):
            u_in = xk.squeeze()

        elif self.model_type == 'corr_only':
            raise Exception('not implemented')
            # u_in = Pyk.squeeze()

        u_in  = np.expand_dims(u_in, axis=0)
        u_in  = self.esn.scaleInput(u_in)
        sk    = self.esn.update(sk, u_in)
        u_out = self.esn.apply(sk, u_in)
        u_out = np.expand_dims(u_out, axis=0)
        yk    = self.esn.unscaleOutput(u_out)

        xk = yk.copy()
        pY = np.zeros((1, self.N_feats_orig))
        pY[0,self.nonzero_ids] = yk

        dec_pred = self.hyperparams['external']['decode_pred']
        if dec_pred:
            assert self.decoder != None, \
                "give a decoder when decode_pred = True"
            assert self.encoder != None, \
                "give an encoder when decode_pred = True"
            assert len(Pxk_dec) > 0, \
                "give a full predictor state"

            full_yk = pY[0,:].reshape(1,self.enclat,
                                      self.enclon,self.filters)
            inputs = [full_yk, Pxk_dec]
            yk_dec = self.decoder.predict(inputs, verbose=0)
            xk_enc = self.encoder.predict(yk_dec, verbose=0)\
                                 .reshape(1, -1, order=self.reshape_order)
            xk[:] = xk_enc[0,self.nonzero_ids]

        return xk, sk, yk

# @keras.saving.register_keras_serializable(name="ESN_embedded")
class ESN_embedded(layers.Layer):
    """ This class is used to embed an ESN into a keras model """

    def __init__(self, esn_params,
                 num_samples=0,
                 **kwargs):

        super(ESN_embedded, self).__init__(**kwargs)

        self.setPars(esn_params, num_samples)

        # part is set here, part externally
        self.esn_ready_to_train = [False, False]
        self.esn_trained = False
        self.last_sk = []

    # def get_config(self):
    #     config = super(ESN_embedded, self).get_config()
    #     config.update({
    #         'esn_params' : keras.saving.serialize_keras_object(self.esn_params)})
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     esn_params_cfg = config.pop('esn_params')
    #     esn_params = keras.saving.deserialize_keras_object(esn_params_cfg)
    #     return cls(mask, **config)

    def setPars(self, pars, num_samples=0):

        self.esn_params = pars
        self.num_samples = num_samples
        self.model_type = pars['external']['model_type']
        self.bypass_mode = pars['external']['bypass_mode']
        self.reshape_order = pars['external']['reshape_order']
        self.needs_initializing = True

    def call(self, inputs, time, control_ft):
        try:
            values = inputs.detach().numpy()
            timeid = time.detach().numpy()[:,0,0,0].astype(int)
            control = control_ft.detach().numpy()

        except TypeError as e:
            return inputs

        if self.needs_initializing:
            self.initialize(values, control)

        if self.bypass_mode:
            return inputs

        self.populate_storage(values, timeid, control)

        if np.all(self.esn_ready_to_train):
            self.train()

        elif self.esn_trained:
            # replace values in inputs with prediction outputs
            outputs = torch.tensor(self.predict(values, timeid, control))
            inputs = ops.where(outputs != np.nan, outputs, inputs)

        return inputs

    def initialize(self, values, control):

        _, self.enclat, self.enclon, self.filters = \
            values.shape

        self.values_dim = self.enclat * self.enclon * self.filters

        _, self.enclat_ct, self.enclon_ct, self.filters_ct = \
            control.shape

        self.control_dim = self.enclat_ct * self.enclon_ct * self.filters_ct

        if self.model_type in ['ESNc', 'DMDc']:
            self.total_feats = self.values_dim + self.control_dim
        else:
            self.total_feats = self.values_dim

        self.storage = np.zeros((self.num_samples, self.total_feats))

        self.populate_lookup = np.zeros((self.num_samples,1))

        print('Initialized embedded ESN')
        if self.bypass_mode: print('Bypass mode: ESN inactive')

        self.needs_initializing = False

    def populate_storage(self, values, timeid, control):
        T, _, _, _ =  values.shape

        # not going to populate storage if timeid beyond range
        if np.any(timeid >= self.storage.shape[0]):
            return

        if not np.all(self.populate_lookup[timeid,:]):
            self.populate_lookup[timeid,:] = 1

            if self.model_type in ['ESNc', 'DMDc']:
                self.storage[timeid, :] = \
                    np.hstack((values.reshape(T, -1,
                                              order=self.reshape_order),
                               control.reshape(T, -1,
                                               order=self.reshape_order)))
            else:
                self.storage[timeid, :] = \
                    values.reshape(T, -1, order=self.reshape_order)

        if np.all(self.populate_lookup):
            # we did the whole epoch, now we can train the ESN
            self.esn_ready_to_train[0] = True
            # reset lookup table for filling the storage in the next
            # epoch
            self.populate_lookup = np.zeros((self.num_samples,1))

    def train(self):
        print('\nTraining embedded ESN')
        Nr = self.esn_params['internal']['Nr']
        Nu = self.total_feats
        Ny = self.values_dim

        # adjust a few parameters based on the model type
        if self.model_type in ['DMD', 'DMDc', 'corr_only']:
            self.esn_params['internal']['dmdMode'] = True
        else:
            self.esn_params['internal']['dmdMode'] = False

        if self.model_type in ['DMD', 'DMDc', 'corr_only', 'ESNc']:
            self.esn_params['internal']['feedThrough'] = True
        else:
            self.esn_params['internal']['feedThrough'] = False

        if self.model_type in ['ESNc']:
            self.esn_params['internal']['ftRange'] = \
                range(self.values_dim,
                      self.total_feats)


        self.esn = ESN(Nr, Nu, Ny)
        self.esn.setPars(self.esn_params['internal'])
        self.esn.initialize()

        # FIXME factorize with rest of ESN interface
        trainU = self.storage[:-1,]
        trainY = self.storage[1:,:self.values_dim]

        self.esn.train(trainU, trainY)

        # reset this flag array
        self.esn_ready_to_train = [False, False]
        self.esn_trained = True

        self.last_sk = self.esn.X[-1,:].copy()

    def predict(self, values, timeid, control):
        outputs = np.zeros_like(values)

        # perform a step for every (value, timeid) pair
        for i, tid in enumerate(timeid):
            outputs[i,:,:,:] = self.step(values[i,:], tid, control[i,:])\
                                   .reshape(self.enclat,
                                            self.enclon,
                                            self.filters)
        return outputs

    def step(self, values, timeid, control):

        # TODO factorize with rest of ESN interface, maybe through a
        # third class that does things.. not sure
        timeid = np.max([1,timeid])

        # get the correct esn state:
        Nt, Nr = self.esn.X.shape
        if np.any(Nt > timeid-1):
            sk = self.esn.X[timeid-1,:]
        elif len(self.last_sk) > 0:
            sk = self.last_sk
        else:
            raise Exception('something is wrong with the ESN states')

        # prepare input data for different model types
        xk = values.reshape(-1, order=self.reshape_order)
        Pxk = control.reshape(-1, order=self.reshape_order)
        if self.model_type in ['ESNc', 'DMDc']:
            u_in = np.append(xk.squeeze(),
                             Pxk.squeeze())
        elif self.model_type in ['ESN', 'DMD']:
            u_in  = xk.squeeze()
        else:
            raise Exception('model type not implemented')

        u_in  = np.expand_dims(u_in, axis=0)
        u_in  = self.esn.scaleInput(u_in)
        sk    = self.esn.update(sk, u_in)
        u_out = self.esn.apply(sk, u_in)
        u_out = np.expand_dims(u_out, axis=0)
        yk    = self.esn.unscaleOutput(u_out).squeeze()
        self.last_sk = sk

        return yk
