import sys
import numpy as np

import keras
import keras_tuner
from keras import layers
from keras import ops
from keras import regularizers
from keras.models import Model

# create custom masking class
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

class AutoEncoder(keras_tuner.HyperModel):

    def __init__(self, test_vec, mask, log_file, esn=None):
        super(AutoEncoder, self).__init__()

        self.test_vec = test_vec
        self.mask = mask
        self.log_file = log_file
        self.esn = esn
        self.resblock_ctr = 0
        self.esn_combine_mode = 'replace'
        self.needs_building = True

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
                    use_timeinput=True,
                    feedthrough_type='multiply',
                    noise_stddev=0.0,
                    dropout_rate=0.0,
                    conv_layers_per_block=3,
                    kernel_size=(3,3),
                    num_filters=32,
                    num_filters_exp=32,
                    num_filters_red=9,
                    inner_stride=1,
                    regularizer=regularizers.L2(1e-5)
                    ):

        self.activation_encoder = 'relu'
        self.activation_decoder = 'relu'
        self.use_feedthrough = use_feedthrough
        self.use_feedthrough_in_esn = use_feedthrough
        self.feedthrough_only = feedthrough_only
        self.feedthrough_type = feedthrough_type
        if self.feedthrough_only: self.use_feedthrough = True

        self.use_timeinput = use_timeinput
        self.noise_stddev = noise_stddev
        self.dropout_rate = dropout_rate
        self.conv_layers_per_block = conv_layers_per_block
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_filters_red = num_filters_red
        self.num_filters_exp = num_filters_exp
        self.regularizer = regularizer
        self.inner_stride = (inner_stride, inner_stride)

        use_dropout = True if self.dropout_rate > 0 else False

        Nlat, Nlon, num_channels = self.test_vec.shape

        masking_layer = Masking(self.mask, name="masking_layer")
        masking_layer_ft = Masking(self.mask, name="masking_layer_ft")

        state_input = layers.Input(shape=(Nlat, Nlon, num_channels),
                                   name="full_state_input")

        if self.use_timeinput:
            time_input = layers.Input(shape=(1,1,1),
                                      name="time_input")

        if self.use_feedthrough:
            feedthrough = layers.Input(shape=(Nlat, Nlon, num_channels),
                                       name="feedthrough_input")

        # Encoder ------------------------------------------------------
        if use_dropout:
            dropout_layer_1 = layers.Dropout(self.dropout_rate,
                                             name="dropout_1")

        conv_block_1 = ConvBlock(self.conv_layers_per_block,
                                 self.num_filters,
                                 self.kernel_size,
                                 self.activation_encoder,
                                 downsample_stride=(2,2),
                                 regularizer=self.regularizer,
                                 name="conv_block_1")

        x = conv_block_1(state_input)

        # second expansion
        conv_block_2 = ConvBlock(self.conv_layers_per_block,
                                 self.num_filters_exp,
                                 self.kernel_size,
                                 self.activation_encoder,
                                 downsample_stride=(2,2),
                                 regularizer=self.regularizer,
                                 name="conv_block_2")

        x = conv_block_2(x)

        # FIXME dropout here?
        if use_dropout: x = dropout_layer_1(x)

        conv_block_3 = ConvBlock(self.conv_layers_per_block,
                                 self.num_filters_red,
                                 self.kernel_size,
                                 self.activation_encoder,
                                 downsample_stride=self.inner_stride,
                                 regularizer=self.regularizer,
                                 name="conv_block_3")

        encoded = conv_block_3(x)

        self.encoder = \
            Model([state_input, time_input], encoded, name="encoder")

        # Call ESN layer in the latent space
        if (self.esn != None):
            # setup feedthrough control
            if self.use_feedthrough_in_esn:
                c = conv_block_1(feedthrough)
                c = conv_block_2(c)

                if use_dropout:
                    c = dropout_layer_1(c)

                control = conv_block_3(c)
            else:
                control = ops.multiply(encoded, 0.0)

            esn_step = self.esn(encoded, time_input, control)

            if (self.esn_combine_mode == 'replace' or
                self.esn.bypass_mode):
                encoded = esn_step
            elif self.esn_combine_mode == 'multiply':
                encoded = layers.Multiply()([esn_step, encoded])
            elif self.esn_combine_mode == 'add':
                encoded = layers.Add()([esn_step, encoded])

        if self.noise_stddev > 0:
            encoded = layers.GaussianNoise(self.noise_stddev)(encoded)

        if use_dropout:
            encoded = layers.Dropout(self.dropout_rate,
                                     name="dropout_1")(encoded)

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
        feedthrough_layer_1 = ConvBlock(self.conv_layers_per_block,
                                        self.num_filters,
                                        self.kernel_size,
                                        strides = (1,1),
                                        activation=self.activation_decoder,
                                        regularizer=self.regularizer,
                                        padding="same",
                                        name='feedthrough_layer_1')

        output_layer = ConvBlock(1, num_channels,
                                 self.kernel_size,
                                 activation="sigmoid",
                                 regularizer=self.regularizer,
                                 name='output_layer')

        # Decoder:
        y = dec_conv_block_1(encoded)
        y = upsample_layer_1(y)
        y = dec_conv_block_2(y)
        y = upsample_layer_2(y)
        y = dec_conv_block_3(y)
        y = upsample_layer_3(y)

        if self.feedthrough_only:
            output = feedthrough_layer_1(feedthrough)
            output = output_layer(output)

            inputs_decoder=[feedthrough]
            inputs_autoencoder=[feedthrough]

        elif self.use_feedthrough:
            z = feedthrough_layer_1(feedthrough)

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
            inputs_decoder=[encoded, feedthrough]
            inputs_autoencoder=[state_input, time_input, feedthrough]

        else:
            output = output_layer(y)
            inputs_decoder=[encoded]
            inputs_autoencoder=[state_input, time_input]

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

    def res_block(self, inputs):
        self.resblock_ctr += 1
        name_conv_a = f"residual_block_conv2d_a_{self.resblock_ctr}"
        name_conv_b = f"residual_block_conv2d_b_{self.resblock_ctr}"
        name_add_layer = f"residual_add_{self.resblock_ctr}"

        x = layers.Conv2D(self.num_filters,
                          self.kernel_size,
                          padding="same",
                          activation="relu",
                          name=name_conv_a)(inputs)
        x = layers.Conv2D(self.num_filters,
                          self.kernel_size,
                          padding="same",
                          name=name_conv_b)(x)
        x = layers.Add(name=name_add_layer)([inputs, x])
        return x


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
                              activity_regularizer=regularizer,
                              padding="same",
                              name=f'{name}_l{ctr}')

            self.layer_list.append(l)

        # final downsampling convolution
        ctr += 1
        l = layers.Conv2D(num_filters,
                          kernel_size,
                          strides = downsample_stride,
                          activation=activation,
                          activity_regularizer=regularizer,
                          padding="same",
                          name=f'{name}_l{ctr}')
        self.layer_list.append(l)

    def __call__(self, inputs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x


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

    def __init__(self, test_data, initial_xk, plotmachine, pars, scalers):
        super().__init__()

        self.initial_xk = initial_xk
        self.test_data = test_data[0]
        self.N_steps = self.test_data.shape[0]
        self.T_test = test_data[1]
        self.test_data_ft = test_data[2]
        self.plotmachine = plotmachine
        self.pars = pars
        self.scalers = scalers
        self.predictions = []
        self.final_error = []
        self.final_base = []

    def on_epoch_end(self, epoch, logs=None):
        self.predictions = np.zeros_like(self.test_data)

        xk   = self.initial_xk[0]
        xkm1 = self.initial_xk[1]
        pb_i = keras.utils.Progbar(self.N_steps,
                                   stateful_metrics=['error', 'base'],
                                   interval=0.5)

        error, base = (0,0)

        for i in range(self.N_steps):

            xk_LR = np.expand_dims(self.test_data_ft[i,], axis=0)

            Pxk = xk_LR

            tid = np.expand_dims(self.T_test[i,], axis=0)
            xkm1 = xk

            if self.pars['feedthrough_only']:
                xk = self.model.predict([Pxk], verbose=0)
            elif self.pars['use_feedthrough']:
                xk = self.model.predict([xk, tid, Pxk], verbose=0)
            else:
                xk = self.model.predict([xk, tid], verbose=0)

            if ('residual_mode' in self.pars and
                self.pars['residual_mode']):
                xk = self.scalers['R']\
                         .inverse_transform(xk.reshape(1,-1))\
                         .reshape(xk.shape) + xk_LR

            self.predictions[i,] = xk
            xk_true = np.expand_dims(self.test_data[i,], axis=0)
            error += (np.sum(np.square(xk - xk_true)))
            base += (np.sum(np.square(xk_LR - xk_true)))
            values = [('error', np.sqrt(error/(i+1))),
                      ('base', np.sqrt(base/(i+1)))]
            pb_i.add(1, values=values)

        self.plotmachine.plot_prediction_error(self.test_data,
                                               self.predictions,
                                               self.test_data_ft,
                                               f'epoch_{epoch}')

        self.final_error = np.sqrt(error/(i+1))
        self.final_base = np.sqrt(base/(i+1))
        logs['error'] = self.final_error
        logs['base']  = self.final_base
