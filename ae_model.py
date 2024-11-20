import sys
import numpy as np

import keras
import keras_tuner
from keras import layers
from keras import ops
from keras import regularizers
from keras.models import Model
from keras.losses import Loss

from compute_tool import ComputeTool
import data_manager as dm

class AutoEncoder(keras_tuner.HyperModel):

    def __init__(
            self,
            **kwargs
    ):
        super(AutoEncoder, self).__init__()

        members_dict = {
            'test_vec' : [],
            'mask' : [],
            'lookback' : 2,
            'learning_rate' : 0.002,
            'optimizer' : 'adam',
            'verbosity' : 20,
            'use_feedthrough' : True,
            'feedthrough_only' : False,
            'feedthrough_type' : 'multiply',
            'multihead_output' : False,
            'noise_stddev' : 0.0,
            'dropout_rate' : 0.0,
            'activation_encoder' : 'leaky_relu',
            'activation_decoder' : 'leaky_relu',
            'num_conv_blocks' : 1,
            'conv_layers_per_block' : 1,
            'num_feedthrough_layers' : 2,
            'num_feedthrough_filters' : 112,
            'num_output_layers' : 2,
            'kernel_size' : (3,3),
            'num_filters' : 32,
            'num_filters_last' : 112,
            'downsample_stride' : (2,2),
            'L2_lambda' : 0.0,
            'RNN_model' : 'RNN',
            'latent_space_dim' : 8,
        }

        # set actual class members
        members_dict.update(kwargs)
        for key, value in members_dict.items():
            setattr(self, key, value)

        self.resblock_ctr = 0
        self.needs_building = True
        self.ct = ComputeTool()
        self.losses = []

        # derived members:
        self.regularizer = regularizers.L2(self.L2_lambda) \
            if self.L2_lambda > 0 else None
        if self.feedthrough_only: self.use_feedthrough = True
        self.use_dropout = True if self.dropout_rate > 0 else False
        # infer dimensions
        self.N_lat, self.N_lon, self.N_chan = self.test_vec.shape
        self.N_lb = self.lookback + 1 # lookback dimension


    def build_model(self):

        masking_layer = Masking(self.mask, name="masking_layer")
        masking_layer_ft = Masking(self.mask, name="masking_layer_ft")

        state_input = layers.Input(shape=(self.N_lb, self.N_lat, self.N_lon,
                                          self.N_chan),
                                   name="full_state_input")

        if self.use_feedthrough:
            feedthrough = layers.Input(
                shape=(self.N_lb, self.N_lat, self.N_lon, self.N_chan),
                name="feedthrough_input")
            ft_inputs = [ops.squeeze(t,axis=1) \
                         for t in ops.split(feedthrough, self.N_lb, axis=1)]

        # Encoder ---------------------
        encoder_dict = self.create_param_dict([
            'num_conv_blocks',
            'conv_layers_per_block',
            'num_filters',
            'num_filters_last',
            'kernel_size',
            'activation_encoder',
            'regularizer',
            'downsample_stride',
        ])

        self.encoding_layers = Encoder(**encoder_dict)

        # split inputs
        state_inputs = [ops.squeeze(t,axis=1) \
                        for t in ops.split(state_input, self.N_lb, axis=1)]

        # apply encoder in training and inference mode separately
        # separate first entry:
        encoded_outputs_0 = \
            self.encoding_layers(state_inputs[0], training=True)

        encoded_outputs_lb = \
            [ self.encoding_layers(inpt, training=False)\
              for inpt in state_inputs[1:] ]

        encoded_outputs = [encoded_outputs_0] + encoded_outputs_lb

        # encoded_outputs = [ self.encoding_layers(inpt) for inpt in state_inputs]
        # encoded_outputs_0 = encoded_outputs[0]

        # apply encoder to feedthrough
        use_encoded_feedthrough = False
        if use_encoded_feedthrough:
            encoded_fts = [ self.encoding_layers(ft, training=False)
                            for ft in ft_inputs]
            encoded_fts = ops.stack(encoded_fts, axis=1)

        # join encoded outputs
        encoded_outputs = ops.stack(encoded_outputs, axis=1)

        self.encoder = \
            Model(state_inputs[0], encoded_outputs_0, name="encoder")

        # Apply noise
        if self.noise_stddev > 0:
            encoded_outputs = \
                layers.GaussianNoise(self.noise_stddev)(encoded_outputs)

        # Apply dropout
        if self.use_dropout:
            encoded_outputs = \
                layers.Dropout(self.dropout_rate)(encoded_outputs)

        if use_encoded_feedthrough:
            encoded_outputs = layers.Concatenate(axis=-1)([encoded_outputs,
                                                           encoded_fts])

        # Run with the RNN
        RNN_dict = self.create_param_dict([
            'RNN_model',
            'activation_encoder',
            'latent_space_dim',
            'num_filters_last'
            ])

        RNN_output = RNNBlock(**RNN_dict)(encoded_outputs)

        # Decoder blocks
        decoder_dict = self.create_param_dict([
            'num_conv_blocks',
            'conv_layers_per_block',
            'num_filters',
            'num_filters_last',
            'kernel_size',
            'activation_decoder',
            'regularizer'
            ])

        self.decoding_layers = Decoder(
            **decoder_dict,
            upsampling_size=self.downsample_stride
        )

        decoded_RNN = self.decoding_layers(RNN_output)
        decoded_AE_only = self.decoding_layers(encoded_outputs_0)

        ds_filters = self.num_feedthrough_filters\
            if self.feedthrough_only else decoded_RNN.shape[-1]
        feedthrough_block = ConvBlock(
            self.num_feedthrough_layers,
            self.num_feedthrough_filters,
            self.kernel_size,
            activation=self.activation_decoder,
            downsample_filters = ds_filters,
            regularizer=self.regularizer,
            name='feedthrough_block'
        )

        output_block = ConvBlock(
            self.num_output_layers,
            self.num_feedthrough_filters,
            self.kernel_size,
            activation=self.activation_decoder,
            downsample_activation="sigmoid",
            downsample_filters=self.N_chan,
            regularizer=self.regularizer,
            name='output_block')

        output_layer_AE_only = ConvBlock(1, self.N_chan,
                                         self.kernel_size,
                                         activation="sigmoid",
                                         regularizer=self.regularizer,
                                         name='output_layer_AE_only')


        if self.feedthrough_only:
            output = feedthrough_block(ft_inputs[0])
            output = output_block(output)
            inputs_full_model=[feedthrough]

        elif self.use_feedthrough:
            output = self.combine_feedthrough(
                decoded_RNN,
                ft_inputs[0],
                self.feedthrough_type,
                feedthrough_block)

            output = output_block(output)
            inputs_full_model=[state_input, feedthrough]

        else:
            output = output_block(decoded_RNN)
            inputs_full_model=[state_input]

        # multiheaded output
        if ( self.multihead_output and
             not self.feedthrough_only ):

            if self.use_feedthrough:
                decoded_AE_only = self.combine_feedthrough(
                    decoded_AE_only,
                    ft_inputs[0],
                    self.feedthrough_type,
                    feedthrough_block)

            #share output layer
            output_AE_only = output_block(decoded_AE_only)

            # different output layer
            # output_AE_only = output_layer_AE_only(decoded_AE_only)
            outputs = [masking_layer(output),
                       masking_layer(output_AE_only),
                       RNN_output]
            self.loss_weights = [1.0, 0.0, 0.0]
        else: # normal output

            outputs = [masking_layer(output)]
            self.loss_weights = None

        # Construct models
        inputs_decoder=[RNN_output]
        outputs_decoder=[decoded_RNN]

        self.decoder = Model(inputs=inputs_decoder,
                             outputs=outputs_decoder,
                             name="decoder")

        self.autoencoder = Model(inputs=inputs_full_model,
                                 outputs=outputs,
                                 name="autoencoder")

        self.log_model()
        self.log_model(self.autoencoder, 'a')

        # models are constructed
        self.needs_building = False

        return self.autoencoder, self.encoder, self.decoder


    def create_param_dict(self, params):
        return \
            {key : self.__dict__[key] for key in params}


    def combine_feedthrough(self, inputs, feedthrough,
                            feedthrough_type='multiply',
                            feedthrough_block=None):

        z = feedthrough_block(feedthrough)
        if feedthrough_type == 'concatenate':
            outputs = layers.Concatenate()([inputs, z])
        elif feedthrough_type == 'multiply':
            # z = keras.activations.sigmoid(z)
            # inputs = keras.activations.sigmoid(inputs)
            outputs = layers.Multiply()([inputs, z])
        elif feedthrough_type == 'ignore':
            outputs = inputs
        else:
            raise Exception('specify feedthrough_type when'
                            ' using feedthrough')
        return outputs


    def compiler(self, model):
        # loss = keras.losses.\
        #     MeanSquaredError(reduction="sum_over_batch_size",
        #                      name="mean_squared_error")

        loss = CustomLoss(losstype='MSE')

        if self.optimizer == 'adam':
            optim = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            optim = keras.optimizers.SGD(learning_rate=self.learning_rate)

        print(f'loss_weights: {self.loss_weights}')
        model.compile(optimizer=optim, loss=loss,
                      loss_weights=self.loss_weights)



    def log_model(self, model=None, mode='a'):
        if model is None:
            model = self
        model.summary()



    def summary(self):

        if self.needs_building:
            print('Model needs building, no summary available.')
            return

        print(f'dropout_rate: {self.dropout_rate}')
        print(f'noise_stddev: {self.noise_stddev}')
        print(f'num_filters: {self.num_filters}')
        print(f'num_filters_last: {self.num_filters_last}')
        print(f'kernel_size: {self.kernel_size}')
        print(f'num_resblocks: {self.num_resblocks}')
        print(f'resblock_ctr: {self.resblock_ctr}')


    def create_unrolled_model(
            self,
            model,
            unroll_dim=0,
            use_clones=False
    ):

        if use_clones:
            cloned_models = [keras.models.clone_model(model)
                             for i in range(unroll_dim)]

            models = [model] + cloned_models
            for i, model in enumerate(models):
                model.name = model.name + f'_{i}'

        if self.feedthrough_only:
            state_input = []
        else:
            state_input = layers.Input(
                shape=(self.N_lb,
                       self.N_lat,
                       self.N_lon,
                       self.N_chan),
                name="unrolled_state_input")

        if self.use_feedthrough:
            feedthrough = [layers.Input(
                shape=(self.N_lb, self.N_lat, self.N_lon, self.N_chan),
                name=f'unrolled_feedthrough_input_{i}')
                           for i in range(unroll_dim+1) ]
        else:
            feedthrough = []

        xk_lb = state_input
        x_out = []
        for i in range(unroll_dim+1):
            model = model if not use_clones else models[i]

            if self.feedthrough_only:
                xk = model(feedthrough[i])
            else:
                xk = model([xk_lb, feedthrough[i]])

            x_out += [xk[0]]

            if unroll_dim > 0:
                # (re)construct lookback array
                xk = ops.expand_dims(xk[0], axis=1)
                xk_lb = ops.concatenate([xk, xk_lb], axis=1)\
                    [:,:self.N_lb,]

        self.unrolled_model = \
            Model(inputs=[state_input]+feedthrough,
                  outputs=x_out,
                  name="unrolled_model")

        self.log_model(self.unrolled_model, 'a')

        self.loss_weights = np.ones(unroll_dim+1) # / np.arange(1,1+unroll_dim+1)
        self.loss_weights = self.loss_weights.tolist()
        self.loss_weights = self.loss_weights[0] \
            if len(self.loss_weights) == 1 else self.loss_weights

        return self.unrolled_model



class Encoder():
    """ Encoder  """

    def __init__(self,
                 num_conv_blocks=3,
                 num_filters=32,
                 num_filters_last=8,
                 conv_layers_per_block=2,
                 kernel_size=(3,3),
                 activation_encoder='relu',
                 regularizer=regularizers.L2(1e-5),
                 downsample_stride=(2,2)
                 ):

        self.block_list = []
        self.x_skip = []

        for i in range(num_conv_blocks):
            nf = num_filters if i < num_conv_blocks - 1 \
                else num_filters_last

            cb = ConvBlock(conv_layers_per_block=conv_layers_per_block,
                           num_filters=nf,
                           kernel_size=kernel_size,
                           activation=activation_encoder,
                           downsample_stride=downsample_stride,
                           regularizer=regularizer,
                           name=f'conv_block_{i+1}')

            self.block_list.append(cb)

    def __call__(self, inputs, **kwargs):
        x = inputs
        for block in self.block_list:
            x = block(x, **kwargs)
            self.x_skip.append(x)
        return x

class Decoder():
    """Decoder: similar to encoder but with upsample layers

        todo: add skip connections if wanted, ConvBlock already
    supports it

    """

    def __init__(self,
                 num_conv_blocks=3,
                 num_filters=32,
                 num_filters_last=8,
                 conv_layers_per_block=2,
                 kernel_size=(3,3),
                 activation_decoder='relu',
                 regularizer=regularizers.L2(1e-5),
                 upsampling_size=(2,2),
                 ):

        self.block_list = []

        for i in range(num_conv_blocks):

            nf = num_filters if i > 0 \
                else num_filters_last

            cb = ConvBlock(conv_layers_per_block=conv_layers_per_block,
                           num_filters=nf,
                           kernel_size=kernel_size,
                           activation=activation_decoder,
                           regularizer=regularizer,
                           name=f'dec_conv_block_{i+1}')
            self.block_list.append(cb)

            ul = layers.UpSampling2D(
                size=upsampling_size,
                interpolation="bilinear"
            )
            self.block_list.append(ul)

        # final_layer = layers.Conv2D(nf,
        #                             kernel_size,
        #                             activation=activation,
        #                             kernel_regularizer=regularizer,
        #                             padding="same",
        #                             name=f'final_dec_conv_layer')
        # self.block_list.append(final_layer)


    def __call__(self, inputs, **kwargs):
        x = inputs
        for block in self.block_list:
            x = block(x, **kwargs)
        return x



class ConvBlock():
    """Convolutional block with optional downsampling stride in the last
    layer.

    """
    def __init__(self,
                 conv_layers_per_block,
                 num_filters=32,
                 kernel_size=(3,3),
                 activation='relu',
                 downsample_stride=(1,1),
                 downsample_filters=None,
                 downsample_activation=None,
                 regularizer=regularizers.L2(1e-5),
                 name="conv_block"):

        self.layer_list = []
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.regularizer = regularizer
        self.activation = activation
        if downsample_filters == None:
            self.downsample_filters = num_filters
        else:
            self.downsample_filters = downsample_filters
        if downsample_activation == None:
            self.downsample_activation = activation
        else:
            self.downsample_activation = downsample_activation

        ctr = 0
        for i in range(conv_layers_per_block-1):
            ctr += 1
            l = layers.Conv2D(num_filters,
                              kernel_size,
                              strides=(1,1),
                              activation=activation,
                              kernel_regularizer=regularizer,
                              padding="same",
                              name=f'{name}_l{ctr}')

            self.layer_list.append(l)

        # final downsampling convolution
        ctr += 1
        l = layers.Conv2D(
            self.downsample_filters,
            kernel_size,
            strides=downsample_stride,
            activation=self.downsample_activation,
            kernel_regularizer=regularizer,
            padding="same",
            name=f'{name}_l{ctr}')

        self.layer_list.append(l)



    def __call__(self, inputs, skip=None, **kwargs):
        if skip == None:
            x = inputs
        else:
            x = layers.Concatenate(axis=-1)([skip, inputs])
        for layer in self.layer_list:
            x = layer(x, **kwargs)
        return x



class RNNBlock():

    def __init__(self,
                 RNN_model='RNN',
                 activation_encoder='relu',
                 latent_space_dim=32,
                 unroll=False,
                 num_filters_last=32,
                 kernel_size=(3,3)):

        self.model = RNN_model
        self.activation = activation_encoder
        self.latent_space_dim = latent_space_dim
        self.filters = num_filters_last
        self.kernel_size = kernel_size
        self.unroll = unroll

    def __call__(self, inputs):
        if self.model == 'RNN':
            return self.RNN(inputs)
        elif self.model == 'RNN_res':
            return self.RNN_res(inputs)
        elif self.model == 'ConvLSTM':
            return self.ConvLSTM(inputs)
        elif self.model == 'GRU':
            return self.GRU(inputs)
        elif self.model == 'LSTM':
            return self.LSTM(inputs)
        elif self.model == 'RNN_var':
            return self.RNN_var(inputs)
        elif self.model == 'disabled':
            return self.most_recent(inputs)
        else:
            raise Exception('Provide a model')


    def ConvLSTM(self, inputs):
        lstm_input = ops.flip(inputs, axis=1)
        lstm_output = \
            layers.ConvLSTM2D(padding='same',
                              filters=self.filters,
                              kernel_size=self.kernel_size)\
                              (lstm_input)
        return lstm_output


    def dense_downsample(
            self,
            inputs
    ):
        x = inputs
        self.Nlb, self.Nj, self.Ni, self.Nc = inputs.shape[1:]
        self.N_feats_in = self.Nj * self.Ni * self.Nc
        self.N_feats_out = self.Nj * self.Ni * self.filters
        x = layers.Reshape((self.Nlb, self.N_feats_in))(x)

        x = layers.Dense(self.latent_space_dim,
                         activation = self.activation)\
                         (x)
        x = ops.flip(x, axis=1)
        return x

    def dense_upsample(
            self,
            inputs
    ):
        x = inputs
        x = layers.Dense(self.N_feats_out,
                         activation = self.activation)\
                         (x)

        return layers.Reshape((self.Nj,
                               self.Ni,
                               self.filters))(x)

    def RNN(self, inputs):
        RNN_input = self.dense_downsample(inputs)
        RNN_output = layers.SimpleRNN(self.latent_space_dim)(RNN_input)
        return self.dense_upsample(RNN_output)

    def RNN_var(self, inputs):

        latent_mean = self.dense_downsample(inputs)
        latent_log_var = self.dense_downsample(inputs)
        sampled = Sampling()(latent_mean, latent_log_var)

        # latent_output = layers.SimpleRNN(self.latent_space_dim)(sampled)
        return self.dense_upsample(sampled[:,0,])

    def GRU(self, inputs):
        GRU_input = self.dense_downsample(inputs)
        GRU_output = layers.GRU(self.latent_space_dim)(GRU_input)
        return self.dense_upsample(GRU_output)

    def LSTM(self, inputs):
        LSTM_input = self.dense_downsample(inputs)
        LSTM_output = layers.LSTM(self.latent_space_dim)(LSTM_input)
        return self.dense_upsample(LSTM_output)

    def RNN_res(self, inputs):
        x = self.most_recent(inputs)
        y = self.RNN(inputs)
        return layers.Add()([x,y])

    def most_recent(self, inputs):
        # assume 5D tensor, time dim ordered from recent to past
        self.N_lb = inputs.shape[1]
        inputs_splitted = \
            [ops.squeeze(t,axis=1) \
             for t in ops.split(inputs, self.N_lb, axis=1)]

        # return most recent time
        return inputs_splitted[0]



class Sampling(layers.Layer):
    """
    Sampling layer
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # optional seed here

    def call(self, mean, log_var):
        btch_dim = ops.shape(mean)[0]
        time_dim = ops.shape(mean)[1]
        feat_dim = ops.shape(mean)[2]
        eps = keras.random.normal(
            shape=(btch_dim, time_dim, feat_dim)
            )
        out = mean + ops.exp(0.5*log_var)*eps
        return out


# custom masking class
@keras.saving.register_keras_serializable(name="custom_masking")
class Masking(layers.Layer):
    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        self.mask = mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'mask' : keras.saving.serialize_keras_object(self.mask)})
        return config

    @classmethod
    def from_config(cls, config):
        mask_config = config.pop("mask")
        mask = keras.saving.deserialize_keras_object(mask_config)
        return cls(mask, **config)

    def call(self, inputs):
        return ops.multiply(inputs, self.mask)

@keras.saving.register_keras_serializable(name="custom_loss")
class CustomLoss(Loss):
    def __init__(
            self,
            name='',
            reduction="sum_over_batch_size",
            losstype='NSE'
    ):
        super().__init__(name=name,
                         reduction=reduction)
        self.losstype = losstype

    def call(self, y_true, y_pred):

        if self.losstype == 'NSE':
            loss = self.normalized_SE(y_true, y_pred)
        elif self.losstype == 'MSE':
            loss = self.mean_SE(y_true, y_pred)

        print(f' :{loss:1.2e}: ', end="")
        return loss

    def normalized_SE(self, y_true, y_pred):
        err = ops.sum(ops.square(y_pred-y_true))
        nrm = ops.sum(ops.square(y_true))
        return (err/nrm)

    def mean_SE(self, y_true, y_pred):
        loss = ops.mean(ops.square(y_pred-y_true))
        return loss


    def get_config(self):
        config = super().get_config()
        return config


class TriggerESN(keras.callbacks.Callback):
    """Callback to control the ESN during training of the AE

    This is very flexible but for now we just want to trigger training
    at the beginning of an epoch and train every x epochs

    Either select a stride <train_every> or train in selected epochs
    <train_in_epochs>.

    """

    def __init__(self, esn, train_every=1,
                 train_in_epochs=[],
                 num_samples=0):
        super().__init__()
        self.esn = esn
        self.train_every = train_every
        self.train_in_epochs = train_in_epochs
        self.num_samples = num_samples

    def on_epoch_begin(self, epoch, logs=None):
        # synchronize the size of the training data between AE and ESN
        # not sure if this is the right place
        self.esn.num_samples = self.num_samples

        if len(self.train_in_epochs) > 0:
            if epoch in self.train_in_epochs:
                self.esn.esn_ready_to_train[1] = True
        elif not epoch % self.train_every:
            self.esn.esn_ready_to_train[1] = True

        if np.all(self.esn.esn_ready_to_train):
            self.esn.train()


class CustomValidation(keras.callbacks.Callback):
    """
    """

    def __init__(self,
                 data,
                 test_inds,
                 plotmachine,
                 pars):

        super().__init__()

        self.pars = pars
        self.unroll_dim = self.pars['unroll_dim']
        self.data = data
        self.test_inds = test_inds

        # account for unrolling
        max_inds = self.data['HR'].shape[0]
        unr_inds = np.arange(test_inds[-1]+1,
                             np.min([test_inds[-1]+self.unroll_dim+1,
                                     max_inds]))
        self.test_inds = np.concatenate([self.test_inds, unr_inds])

        self.test_data = self.data['HR'][test_inds,]
        self.test_data_ft = self.data['LR'][test_inds,]
        self.N_steps = self.test_data.shape[0]
        self.plotmachine = plotmachine
        self.predictions = []
        self.final_error = []
        self.final_base = []
        self.lookback = self.pars['lookback']

    def on_epoch_begin(self, epoch, logs=None):
        if self.pars['predict_only']:
            self.predict(epoch, logs)
            self.model.stop_training=True

    def on_epoch_end(self, epoch, logs=None):
        if not self.pars['predict_only']:
            self.predict(epoch, logs)

    def predict(self, epoch, logs=None):
        self.predictions = np.zeros_like(self.test_data)

        init_ind = self.test_inds[0]-1

        xk_lb = np.expand_dims(
            dm.create_lookback(init_ind, [self.data['HR']],
                               self.lookback,axis=0)[0], axis=0)

        pb_i = keras.utils.Progbar(self.N_steps,
                                   stateful_metrics=['error', 'base'],
                                   interval=0.5)
        error, base = (0,0)

        for i in range(self.N_steps-self.unroll_dim):


            xk_LR = [ np.expand_dims(
                dm.create_lookback(self.test_inds[i+unroll], [self.data['LR']],
                                   self.lookback, axis=0)[0], axis=0)
                      for unroll in range(self.unroll_dim+1) ]

            Pxk = xk_LR

            if self.pars['feedthrough_only']:
                xk = self.model.predict([Pxk[0]], verbose=0)
            elif self.pars['use_feedthrough']:
                xk = self.model.predict([xk_lb]+Pxk, verbose=0)
            else:
                xk = self.model.predict([xk_lb], verbose=0)

            # if ( self.pars['multihead_output'] and
            #      not self.pars['feedthrough_only'] ): xk = xk[0]

            if ( isinstance(xk, list) and
                 self.unroll_dim > 0 ):
                for j in range(self.unroll_dim+1):
                    self.predictions[i+j,] = xk[j]
                xk = xk[0]
            else:
                self.predictions[i,] = xk

            if self.pars['evaluate']:
                xk_true = np.expand_dims(self.test_data[i,], axis=0)
                error += (np.sum(np.square(xk - xk_true)))
                base += (np.sum(np.square(xk_LR[0][:,0,] - xk_true)))
                values = [('error', np.sqrt(error/(i+1))),
                          ('base', np.sqrt(base/(i+1)))]

                pb_i.add(1, values=values)
            else:
                pb_i.add(1)

            xk = np.expand_dims(xk, axis=1)
            xk_lb = np.concatenate([xk, xk_lb], axis=1)\
                [:,:self.lookback+1,]


        if self.pars['evaluate']:
            self.plotmachine.plot_prediction_error(self.test_data,
                                                   self.predictions,
                                                   self.test_data_ft,
                                                   f'epoch_{epoch}')

            self.final_error = np.sqrt(error/(i+1))
            self.final_base = np.sqrt(base/(i+1))
            logs['error'] = self.final_error
            logs['base']  = self.final_base
