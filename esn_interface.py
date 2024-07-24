from ESN.ESN import ESN

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
                                self.xLR_train[-self.history+1:,] * self.control_amp))

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
