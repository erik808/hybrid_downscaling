import sys
import numpy as np

import keras
import keras_tuner
from keras import layers
from keras import ops
from keras import regularizers
from keras.models import Model

from compute_tool import ComputeTool
import data_manager as dm

class AutoEncoder(keras_tuner.HyperModel):

    def __init__(self, test_vec,
                 mask, log_file,
                 esn=None, lookback=0):
        super(AutoEncoder, self).__init__()

        self.test_vec = test_vec
        self.mask = mask
        self.log_file = log_file
        self.esn = esn
        self.lookback = lookback
        self.resblock_ctr = 0
        self.esn_combine_mode = 'replace'
        self.needs_building = True
        self.ct = ComputeTool()

        self.log('AutoEncoder\n', 'w')

    def summary(self):

        if self.needs_building:
            print('Model needs building, no summary available.')
            return

        print(f'dropout_rate: {self.dropout_rate}')
        print(f'noise_stddev: {self.noise_stddev}')
        print(f'esn: {vars(self.esn)}')
        print(f'num_filters: {self.num_filters}')
        print(f'num_filters_red: {self.num_filters_red}')
        print(f'kernel_size: {self.kernel_size}')
        print(f'num_resblocks: {self.num_resblocks}')
        print(f'resblock_ctr: {self.resblock_ctr}')
        print(f'esn_combine_mode: {self.esn_combine_mode}')


    def build_model(self,
                    conv_arch='default',
                    learning_rate=0.002,
                    optimizer='adam',
                    verbosity=20,
                    use_feedthrough=True,
                    feedthrough_only=False,
                    feedthrough_type='multiply',
                    noise_stddev=0.0,
                    dropout_rate=0.0,
                    conv_layers_per_block=1,
                    num_feedthrough_layers=2,
                    kernel_size=(3,3),
                    num_filters=32,
                    num_filters_exp=32,
                    num_filters_red=9,
                    inner_stride=1,
                    L2_lambda=1e-5,
                    ):

        self.activation_encoder = 'relu'
        self.activation_decoder = 'relu'
        self.use_feedthrough = use_feedthrough
        self.use_feedthrough_in_esn = use_feedthrough
        self.feedthrough_only = feedthrough_only
        self.feedthrough_type = feedthrough_type
        if self.feedthrough_only: self.use_feedthrough = True

        self.noise_stddev = noise_stddev
        self.dropout_rate = dropout_rate
        self.conv_layers_per_block = conv_layers_per_block
        self.num_feedthrough_layers = num_feedthrough_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_filters_red = num_filters_red
        self.num_filters_exp = num_filters_exp
        self.regularizer = regularizers.L2(L2_lambda) \
            if L2_lambda > 0 else None
        self.inner_stride = (inner_stride, inner_stride)

        use_dropout = True if self.dropout_rate > 0 else False

        # infer dimensions
        Nlat, Nlon, num_channels = self.test_vec.shape
        N_lb = self.lookback + 1 # lookback dimension

        masking_layer = Masking(self.mask, name="masking_layer")
        masking_layer_ft = Masking(self.mask, name="masking_layer_ft")

        state_input = layers.Input(shape=(N_lb, Nlat, Nlon,
                                          num_channels),
                                   name="full_state_input")



        if self.use_feedthrough:
            feedthrough = layers.Input(shape=(N_lb, Nlat, Nlon, num_channels),
                                       name="feedthrough_input")
            ft_inputs = [ops.squeeze(t,axis=1) \
                         for t in ops.split(feedthrough, N_lb, axis=1)]

        # Encoder ------------------------------------------------------
        if use_dropout:
            dropout_layer_1 = layers.Dropout(self.dropout_rate,
                                             name="dropout_1")

        encoding_layers = Encoder(
            conv_layers_per_block=self.conv_layers_per_block,
            num_filters=[self.num_filters,
                         self.num_filters_exp, # expansion
                         self.num_filters_red, # reduction
                         ],
            kernel_size=self.kernel_size,
            activation=self.activation_encoder,
            downsample_strides=[(2,2),
                                (2,2),
                                self.inner_stride],
            regularizer=self.regularizer)

        # split inputs
        state_inputs = [ops.squeeze(t,axis=1) \
                        for t in ops.split(state_input, N_lb, axis=1)]

        # apply encoder separately
        encoded_outputs = [ encoding_layers(inpt) for inpt in state_inputs]
        encoded_ft = encoding_layers(ft_inputs[0])

        # join encoded outputs
        encoded_outputs = ops.stack(encoded_outputs, axis=1)

        self.encoder = \
            Model(state_inputs[0], encoded_outputs[0], name="encoder")

        # !!! THIS WAY OF CALLING THE ESN LAYER IS DEPRECATED !!!
        # Call ESN layer in the latent space
        # if (self.esn != None):
        #     # setup feedthrough control
        #     if self.use_feedthrough_in_esn:
        #         control = encoding_layers(feedthrough)
        #     else:
        #         control = ops.multiply(encoded, 0.0)

            # esn_step = self.esn(encoded, time_input, control)

            # if (self.esn_combine_mode == 'replace' or
            #     self.esn.bypass_mode):
            #     encoded = esn_step
            # elif self.esn_combine_mode == 'multiply':
            #     encoded = layers.Multiply()([esn_step, encoded])
            # elif self.esn_combine_mode == 'add':
            #     encoded = layers.Add()([esn_step, encoded])

        # Apply noise
        if self.noise_stddev > 0:
            encoded_outputs = \
                layers.GaussianNoise(self.noise_stddev)(encoded_outputs)

        # Apply dropout
        if use_dropout:
            encoded_outputs = \
                layers.Dropout(self.dropout_rate)(encoded_outputs)

        # Run with the RNN
        RNN_output = RNNBlock(model='RNN',
                              activation=self.activation_encoder,
                              reduction_factor=self.num_filters_red)\
                              (encoded_outputs)

        # use_encoded_feedthrough = True
        # if use_encoded_feedthrough:
        #     RNN_output = layers.Multiply()([RNN_output, encoded_ft])

        # Decoder blocks
        dec_conv_block_1 = ConvBlock(self.conv_layers_per_block,
                                     self.num_filters,
                                     self.kernel_size,
                                     self.activation_decoder,
                                     regularizer=self.regularizer,
                                     name="dec_conv_block_1")

        upsample_layer_1 = layers.UpSampling2D(size=self.inner_stride,
                                               interpolation="bilinear")

        dec_conv_block_2 = ConvBlock(self.conv_layers_per_block,
                                     self.num_filters_exp,
                                     self.kernel_size,
                                     self.activation_decoder,
                                     regularizer=self.regularizer,
                                     name="dec_conv_block_2")

        upsample_layer_2 = layers.UpSampling2D(size=(2, 2),
                                               interpolation="bilinear")

        dec_conv_block_3 = ConvBlock(self.conv_layers_per_block,
                                     self.num_filters,
                                     self.kernel_size,
                                     self.activation_decoder,
                                     regularizer=self.regularizer,
                                     name="dec_conv_block_3")

        upsample_layer_3 = layers.UpSampling2D(size=(2,2),
                                               interpolation="bilinear")

        dropout_layer_2 = layers.Dropout(self.dropout_rate,
                                         name="dropout_2")

        # Should these be residual blocks instead?
        feedthrough_block = ConvBlock(self.num_feedthrough_layers,
                                      self.num_filters,
                                      self.kernel_size,
                                      activation=self.activation_decoder,
                                      regularizer=self.regularizer,
                                      name='feedthrough_block')

        output_layer = ConvBlock(1, num_channels,
                                 self.kernel_size,
                                 activation="sigmoid",
                                 regularizer=self.regularizer,
                                 name='output_layer')


        y = dec_conv_block_1(RNN_output)
        y = upsample_layer_1(y)
        y = dec_conv_block_2(y)
        y = upsample_layer_2(y)
        y = dec_conv_block_3(y)
        y = upsample_layer_3(y)

        if self.feedthrough_only:
            output = feedthrough_block(ft_inputs[0])
            output = output_layer(output)

            inputs_decoder=[feedthrough]
            inputs_autoencoder=[feedthrough]

        elif self.use_feedthrough:
            z = feedthrough_block(ft_inputs[0])

            if feedthrough_type == 'concatenate':
                output = layers.Concatenate()([y, z])
            elif feedthrough_type == 'multiply':
                output = layers.Multiply()([y, z])
            elif feedthrough_type == 'ignore':
                output = y
            else:
                raise Exception('specify feedthrough_type when'
                                ' using feedthrough')

            output = output_layer(output)
            inputs_decoder=[RNN_output, feedthrough]
            inputs_autoencoder=[state_input, feedthrough]

        else:
            output = output_layer(y)
            inputs_decoder=[RNN_output]
            inputs_autoencoder=[state_input]

        outputs = [masking_layer(output)]

        # Construct models
        self.decoder = Model(inputs=inputs_decoder,
                             outputs=outputs,
                             name="decoder")

        self.autoencoder = Model(inputs=inputs_autoencoder,
                            outputs=outputs,
                            name="autoencoder")

        loss = keras.losses.\
            MeanSquaredError(reduction="sum_over_batch_size",
                             name="mean_squared_error")


        if optimizer == 'adam':
            optim = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optim = keras.optimizers.SGD(learning_rate=learning_rate)

        self.autoencoder.compile(optimizer=optim, loss=loss)

        self.log_model()
        self.log_model(self.autoencoder, 'a')
        self.log_model(self.esn, 'a')

        # models are constructed
        self.needs_building = False

        return self.autoencoder, self.encoder, self.decoder

    # build model for hyperparameter tuning
    def build(self, hp):
        learning_rate = hp.Float("learning_rate",
                                 min_value=1e-4,
                                 max_value=5e-2,
                                 sampling="log")
        hypermodel, _, _ = self.build_model(learning_rate=learning_rate)
        return hypermodel

    def fit(self, hp, model, *args, **kwargs):
        batch_sizes = hp.Int("batch_size",
                             min_value=2,
                             max_value=32)
        return model.fit(*args, batch_size=batch_sizes,
                         **kwargs)

    def log(self, msg, mode='a'):
        original = sys.stdout
        with open(self.log_file, mode) as f:
            sys.stdout = f
            print(msg)
            sys.stdout = original

    def log_model(self, model=None, mode='a'):
        if model is None:
            model = self

        original = sys.stdout
        with open(self.log_file, mode) as f:
            sys.stdout = f
            model.summary()
            sys.stdout = original


class Encoder():
    """Encoder. For now hardcoded to contain three convolutional
    blocks. Hardcoding can be dealt with later.

    """

    def __init__(self,
                 num_conv_blocks=3,
                 conv_layers_per_block=2,
                 num_filters=[32,32,8],
                 kernel_size=(3,3),
                 activation='relu',
                 downsample_strides=[(2,2), (2,2), (1,1)],
                 regularizer=regularizers.L2(1e-5)
                 ):


        self.block_list = []
        self.x_skip = []

        for i in range(num_conv_blocks):
            cb = ConvBlock(conv_layers_per_block=conv_layers_per_block,
                           num_filters=num_filters[i],
                           kernel_size=kernel_size,
                           activation=activation,
                           downsample_stride=downsample_strides[i],
                           regularizer=regularizer,
                           name=f'conv_block_{i+1}')

            self.block_list.append(cb)

    def __call__(self, inputs):
        x = inputs
        for block in self.block_list:
            x = block(x)
            self.x_skip.append(x)

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
                 regularizer=regularizers.L2(1e-5),
                 name="conv_block"):

        self.layer_list = []
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
        l = layers.Conv2D(num_filters,
                          kernel_size,
                          strides = downsample_stride,
                          activation=activation,
                          kernel_regularizer=regularizer,
                          padding="same",
                          name=f'{name}_l{ctr}')
        self.layer_list.append(l)

    def __call__(self, inputs, skip=None):
        if skip == None:
            x = inputs
        else:
            x = layers.Concatenate(axis=-1)([skip, inputs])
        for layer in self.layer_list:
            x = layer(x)
        return x



class RNNBlock():

    def __init__(self,
                 model='RNN',
                 activation='relu',
                 reduction_factor=1,
                 unroll=False,
                 filters=32,
                 kernel_size=(3,3)):

        self.model = model
        self.activation = activation
        self.reduction_factor = reduction_factor
        self.filters = filters
        self.kernel_size = kernel_size
        self.unroll = unroll

    def __call__(self, inputs):
        if self.model == 'RNN':
            return self.RNN(inputs)
        elif self.model == 'RNN_res':
            return self.RNN_res(inputs)
        elif self.model == 'ConvLSTM':
            return self.ConvLSTM(inputs)
        else:
            return self.most_recent(inputs)

    def ConvLSTM(self, inputs):
        lstm_input = ops.flip(inputs, axis=1)
        lstm_output = \
            layers.ConvLSTM2D(padding='same',
                              filters=self.filters,
                              kernel_size=self.kernel_size)\
                              (lstm_input)
        return lstm_output

    def RNN(self, inputs):

        Nlb, Nj, Ni, Nc = inputs.shape[1:]
        N_feats = Nj * Ni * Nc
        RNN_rdim = N_feats // self.reduction_factor
        RNN_input = layers.Reshape((Nlb, N_feats))(inputs)
        RNN_input = layers.Dense(RNN_rdim,
                                 activation = self.activation)(RNN_input)
        RNN_input = ops.flip(RNN_input, axis=1)
        RNN_output = layers.SimpleRNN(RNN_rdim)(RNN_input)
        RNN_output = layers.Dense(N_feats,
                                  activation = self.activation)(RNN_output)
        return layers.Reshape((Nj, Ni, Nc))(RNN_output)

    def RNN_res(self, inputs):
        x = self.most_recent(inputs)
        y = self.RNN(inputs)
        return layers.Add()([x,y])

    def most_recent(self, inputs):
        # assume 5D tensor, time dim ordered from recent to past
        N_lb = inputs.shape[1]
        inputs_splitted = \
            [ops.squeeze(t,axis=1) \
             for t in ops.split(inputs, N_lb, axis=1)]

        # return most recent time
        return inputs_splitted[0]



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

        self.data = data
        self.test_inds = test_inds
        self.test_data = self.data['HR'][test_inds,]
        self.test_data_ft = self.data['LR'][test_inds,]
        self.N_steps = self.test_data.shape[0]
        self.plotmachine = plotmachine
        self.pars = pars
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

        for i in range(self.N_steps):

            xk_LR = np.expand_dims(
                dm.create_lookback(self.test_inds[i], [self.data['LR']],
                                   self.lookback, axis=0)[0], axis=0)
            Pxk = xk_LR

            if self.pars['feedthrough_only']:
                xk = self.model.predict([Pxk], verbose=0)
            elif self.pars['use_feedthrough']:
                xk = self.model.predict([xk_lb, Pxk], verbose=0)
            else:
                xk = self.model.predict([xk_lb], verbose=0)

            self.predictions[i,] = xk

            if self.pars['evaluate']:

                xk_true = np.expand_dims(self.test_data[i,], axis=0)
                error += (np.sum(np.square(xk - xk_true)))
                base += (np.sum(np.square(xk_LR[:,0,] - xk_true)))
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
