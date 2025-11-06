simport numpy as np

import torch
import keras
from keras import layers
from keras import ops
from keras import regularizers
from keras.models import Model
from keras.losses import Loss

import data_utils


class AutoEncoder():

    def __init__(
            self,
            **kwargs
    ):
        members_dict = {
            'test_vec': [],
            'mask': [],
            'lookback': 2,
            'learning_rate': 0.002,
            'optimizer': 'adam',
            'verbosity': 20,
            'use_feedthrough': True,
            'feedthrough_only': False,
            'feedthrough_type': 'multiply',
            'noise_stddev': 0.0,
            'dropout_rate': 0.0,
            'activation_encoder': 'leaky_relu',
            'activation_decoder': 'leaky_relu',
            'num_conv_blocks': 1,
            'conv_layers_per_block': 1,
            'num_feedthrough_layers': 2,
            'num_feedthrough_filters': 112,
            'num_output_layers': 2,
            'kernel_size': (3, 3),
            'num_filters': 32,
            'num_filters_last': 112,
            'downsample_stride': (2, 2),
            'L2_lambda': 0.0,
            'latent_space_model': 'RNN',
            'latent_space_dim': 8,
        }

        #  set actual class members
        members_dict.update(kwargs)
        for key, value in members_dict.items():
            setattr(self, key, value)

        self.resblock_ctr = 0
        self.needs_building = True
        self.losses = []

        # derived members:
        self.regularizer = None  # used to be derived from L2_lambda
        if self.feedthrough_only:
            self.use_feedthrough = True
        self.use_dropout = True if self.dropout_rate > 0 else False
        # infer dimensions
        self.N_lat, self.N_lon, self.N_chan = self.test_vec.shape
        self.N_lb = self.lookback + 1  # lookback dimension
        self.loss_weights = None

    def build_model(self):

        masking_layer = Masking(self.mask, name="masking_layer")

        # input and feedthrough dimensions:
        # <batch_size> x <lookback> x <lat> x <lon> x <channels>
        # lookback axis is backward in time:
        # so input[:,0,] is the most recent field
        state_input = layers.Input(
            shape=(self.N_lb,
                   self.N_lat,
                   self.N_lon,
                   self.N_chan),
            name="full_state_input")

        if self.use_feedthrough:
            feedthrough = layers.Input(
                shape=(self.N_lb, self.N_lat, self.N_lon, self.N_chan),
                name="feedthrough_input")
            ft_inputs = [ops.squeeze(t, axis=1)
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

        # split inputs along time axis (axis=1)
        state_inputs = [ops.squeeze(t, axis=1)
                        for t in ops.split(state_input, self.N_lb, axis=1)]

        # apply encoder in training and inference mode separately
        # separate first entry:
        encoded_outputs_0 = \
            self.encoding_layers(state_inputs[0], training=True)

        encoded_outputs_lb = \
            [self.encoding_layers(inpt, training=False)
             for inpt in state_inputs[1:]]

        encoded_outputs = [encoded_outputs_0] + encoded_outputs_lb

        # encoded_outputs = \
        # [ self.encoding_layers(inpt) for inpt in state_inputs]
        # encoded_outputs_0 = encoded_outputs[0]

        # apply encoder to feedthrough
        # use_encoded_feedthrough = False
        # if use_encoded_feedthrough:
        #     encoded_fts = [ self.encoding_layers(ft, training=False)
        #                     for ft in ft_inputs]
        #     encoded_fts = ops.stack(encoded_fts, axis=1)

        # join encoded outputs
        encoded_outputs = ops.stack(encoded_outputs, axis=1)

        # Apply noise
        if self.noise_stddev > 0:
            encoded_outputs = \
                layers.GaussianNoise(self.noise_stddev)(encoded_outputs)

        # Apply dropout
        if self.use_dropout:
            encoded_outputs = \
                layers.Dropout(self.dropout_rate)(encoded_outputs)

        # if use_encoded_feedthrough:
        #     encoded_outputs = layers.Concatenate(axis=-1)([encoded_outputs,
        #                                                    encoded_fts])

        # Create and call model in the latent space
        lspacemod_dict = \
            self.create_param_dict([
                'latent_space_model',
                'activation_encoder',
                'latent_space_dim',
                'num_filters_last'
            ])

        lspace_model = LatentSpaceModel(**lspacemod_dict)
        lspace_model_output = lspace_model(encoded_outputs)
        lspace_vars = lspace_model.get_lspace_vars()

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

        decoded_state = self.decoding_layers(lspace_model_output)

        ds_filters = self.num_feedthrough_filters\
            if self.feedthrough_only else decoded_state.shape[-1]

        feedthrough_block = ConvBlock(
            self.num_feedthrough_layers,
            self.num_feedthrough_filters,
            self.kernel_size,
            activation=self.activation_decoder,
            downsample_filters=ds_filters,
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

        if self.feedthrough_only:
            output = feedthrough_block(ft_inputs[0])
            output = output_block(output)
            inputs_full_model=[feedthrough]
            lspace_model_output=state_input
            inputs_decoder=[lspace_model_output, feedthrough]

        elif self.use_feedthrough:
            output = self.combine_feedthrough(
                decoded_state,
                ft_inputs[0],
                self.feedthrough_type,
                feedthrough_block)

            output = output_block(output)
            inputs_full_model=[state_input, feedthrough]
            inputs_decoder=[lspace_model_output, feedthrough]

        else:
            output = output_block(decoded_state)
            inputs_full_model=[state_input]
            inputs_decoder=[lspace_model_output]

        outputs = [masking_layer(output)]
        # outputs = output

        # Create encoder
        self.encoder = Model(
            inputs=state_input,
            outputs=[lspace_model_output, lspace_vars],
            name="encoder")

        self.decoder = Model(inputs=inputs_decoder,
                             outputs=outputs,
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
            {key: self.__dict__[key] for key in params}

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
        loss = None
        if self.optimizer == 'adam':
            optim = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            optim = keras.optimizers.SGD(learning_rate=self.learning_rate)

        print(f'loss_weights: {self.loss_weights}')
        model.compile(optimizer=optim,
                      loss=loss,
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
            feedthrough = [
                layers.Input(
                    shape=(self.N_lb, self.N_lat, self.N_lon, self.N_chan),
                    name=f'unrolled_feedthrough_input_{i}')
                for i in range(unroll_dim + 1)
            ]
        else:
            feedthrough = []

        xk_lb = state_input
        x_out = []
        for i in range(unroll_dim + 1):
            model = model if not use_clones else models[i]

            if self.feedthrough_only:
                xk = model(feedthrough[i])
            else:
                xk = model([xk_lb, feedthrough[i]])

            x_out += [xk[0]]

            if unroll_dim > 0:
                # (re)construct lookback array
                xk = ops.expand_dims(xk[0], axis=1)
                xk_lb = (ops.concatenate([xk, xk_lb], axis=1)
                         [:, :self.N_lb,])

        self.unrolled_model = \
            Model(inputs=[state_input] + feedthrough,
                  outputs=x_out,
                  name="unrolled_model")

        self.log_model(self.unrolled_model, 'a')

        self.loss_weights = np.ones(unroll_dim + 1)
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
                 kernel_size=(3, 3),
                 activation_encoder='relu',
                 regularizer=regularizers.L2(1e-5),
                 downsample_stride=(2, 2)
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
                 kernel_size=(3, 3),
                 activation_decoder='relu',
                 regularizer=regularizers.L2(1e-5),
                 upsampling_size=(2, 2),
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
                 kernel_size=(3, 3),
                 activation='relu',
                 downsample_stride=(1, 1),
                 downsample_filters=None,
                 downsample_activation=None,
                 regularizer=regularizers.L2(1e-5),
                 name="conv_block"):

        self.layer_list = []
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.regularizer = regularizer
        self.activation = activation
        if downsample_filters is None:
            self.downsample_filters = num_filters
        else:
            self.downsample_filters = downsample_filters
        if downsample_activation is None:
            self.downsample_activation = activation
        else:
            self.downsample_activation = downsample_activation

        ctr = 0
        for i in range(conv_layers_per_block - 1):
            ctr += 1
            conv_l = layers.Conv2D(num_filters,
                                   kernel_size,
                                   strides=(1, 1),
                                   activation=activation,
                                   kernel_regularizer=regularizer,
                                   padding="same",
                                   name=f'{name}_l{ctr}')

            self.layer_list.append(conv_l)

        # final downsampling convolution
        ctr += 1
        conv_l = layers.Conv2D(
            self.downsample_filters,
            kernel_size,
            strides=downsample_stride,
            activation=self.downsample_activation,
            kernel_regularizer=regularizer,
            padding="same",
            name=f'{name}_l{ctr}')

        self.layer_list.append(conv_l)



    def __call__(self, inputs, skip=None, **kwargs):
        if skip == None:
            x = inputs
        else:
            x = layers.Concatenate(axis=-1)([skip, inputs])
        for layer in self.layer_list:
            x = layer(x, **kwargs)
        return x


class LatentSpaceModel():

    def __init__(self,
                 latent_space_model='RNN',
                 activation_encoder='relu',
                 latent_space_dim=32,
                 unroll=False,
                 num_filters_last=32,
                 kernel_size=(3, 3)):

        self.model = latent_space_model
        self.activation = activation_encoder
        self.latent_space_dim = latent_space_dim
        self.filters = num_filters_last
        self.kernel_size = kernel_size
        self.unroll = unroll
        self.lspace_vars = None
        self.RNN_pars = {
            'units': self.latent_space_dim,
            'return_sequences': True,
            'return_state': False,
            # 'go_backwards': True,
        }

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
        elif self.model == 'AE':
            return self.most_recent(inputs)
        elif self.model == 'VAE':
            return self.VAE(inputs)
        elif self.model == 'VAE+RNN':
            return self.VAE(inputs,
                            latent_RNN=True)
        elif self.model == 'disabled':
            return self.most_recent(inputs)
        else:
            raise Exception('Provide a valid latent space model')

    def ConvLSTM(self, inputs):
        lstm_input = ops.flip(inputs, axis=1)
        lstm_output = \
            layers.ConvLSTM2D(padding='same',
                              filters=self.filters,
                              kernel_size=self.kernel_size)(lstm_input)
        return lstm_output

    def dense_downsample(
            self,
            inputs,
            activation='class_default',
            do_flip=True,
            **kwargs,
    ):
        activ = self.activation if activation == 'class_default' \
            else activation

        self.Nlb, self.Nj, self.Ni, self.Nc = inputs.shape[1:]
        self.N_feats_in = self.Nj * self.Ni * self.Nc
        self.N_feats_out = self.Nj * self.Ni * self.filters

        x = layers.Reshape((self.Nlb, self.N_feats_in))(inputs)
        x = layers.Dense(
            self.latent_space_dim,
            activation=activ,
            **kwargs,
        )(x)

        if do_flip:
            x = ops.flip(x, axis=1)
        return x

    def dense_upsample(
            self,
            inputs,
            activation='class_default',
            **kwargs,
    ):
        activ = self.activation if activation == 'class_default' \
            else activation

        x = inputs
        x = layers.Dense(
            self.N_feats_out,
            activation=activ,
            **kwargs,
        )(x)

        return layers.Reshape((self.Nj,
                               self.Ni,
                               self.filters))(x)

    def RNN(self, inputs):
        RNN_input = self.dense_downsample(inputs)
        RNN_output = layers.SimpleRNN(**self.RNN_pars)(RNN_input)

        # store these vars as lspace vars
        self.lspace_vars = {
            'rnn_input': RNN_input,
            'rnn_output': RNN_output
        }
        return self.dense_upsample(RNN_output[:, -1,])

    def VAE(self, inputs, latent_RNN=False):

        downsampled = self.dense_downsample(inputs)

        # note these are flipped in time
        mean = layers.Dense(self.latent_space_dim)(downsampled)
        log_var = layers.Dense(self.latent_space_dim)(downsampled)

        self.lspace_vars = {
            'mean': mean,
            'log_var': log_var
        }

        if latent_RNN:
            RNN_pars = {
                'units': self.latent_space_dim,
                # 'activation': 'sigmoid',
                'return_sequences': True,
                'return_state': False
            }
            rnn_mean = (layers.SimpleRNN(**self.RNN_pars)
                        (ops.flip(mean, axis=1))
                        )

            rnn_log_var = (layers.SimpleRNN(**RNN_pars)
                           (ops.flip(log_var, axis=1))
                           )

            self.lspace_vars.update({
                'rnn_mean': rnn_mean,
                'rnn_log_var': rnn_log_var,
            })

            sampled = Sampling()(rnn_mean[:, -1,],
                                 rnn_log_var[:, -1,])
            out = self.dense_upsample(sampled)

        else:
            # ordering in time here is last last
            sampled = Sampling()(mean[:, -1,], log_var[:, -1,])
            out = self.dense_upsample(sampled)

        return out

    def GRU(self, inputs):
        GRU_input = self.dense_downsample(inputs)
        GRU_output = layers.GRU(**self.RNN_pars)(GRU_input)
        return self.dense_upsample(GRU_output)

    def LSTM(self, inputs):
        LSTM_input = self.dense_downsample(inputs)
        LSTM_output = layers.LSTM(**self.RNN_pars)(LSTM_input)
        # store these vars as lspace vars
        self.lspace_vars = {
            'rnn_input': LSTM_input,
            'rnn_output': LSTM_output
        }

        return self.dense_upsample(LSTM_output[:, -1,])

    def RNN_res(self, inputs):
        x = self.most_recent(inputs)
        y = self.RNN(inputs)
        return layers.Add()([x, y])

    def most_recent(self, inputs):
        # assume 5D tensor, time dim ordered from recent to past
        self.N_lb = inputs.shape[1]
        inputs_splitted = \
            [ops.squeeze(t, axis=1)
             for t in ops.split(inputs, self.N_lb, axis=1)]

        # return most recent time
        return inputs_splitted[0]

    def get_lspace_vars(self):
        if self.lspace_vars is None:
            return []
        else:
            return self.lspace_vars


class LSModelWrapper(keras.Model):

    def __init__(
            self,
            encoder,
            decoder,
            model='RNN',
            **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.model = model
        self.total_loss_tracker = \
            keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = \
            keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = \
            keras.metrics.Mean(name="KL_loss")
        self.rnn_loss_tracker = \
            keras.metrics.Mean(name="rnn_loss")
        self.loss_fn = keras.losses.MeanSquaredError()
        # self.loss_fn = CustomLoss(losstype='MSE')

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.rnn_loss_tracker,
                ]

    def call(self, inputs):
        ft_mode = (len(inputs) == 2)
        enc_output, z_vars = self.encoder(inputs[0])
        if ft_mode:
            input_decoder = [enc_output, inputs[1]]
        else:
            input_decoder = enc_output
        pred = self.decoder(input_decoder)
        return pred

    def train_step(self, data):
        # common
        self.zero_grad()

        if self.model == 'VAE':
            return self.train_step_VAE(data)
        elif self.model == 'VAE+RNN':
            return self.train_step_VAE(data,
                                       RNN_hybrid=True)
        elif self.model in ['RNN',
                            'LSTM',
                            'GRU']:
            return self.train_step_RNN(data)
        elif self.model in ['AE',
                            'disabled']:
            return self.train_step_AE(data)
        else:
            raise Exception('no valid latent space model defined')

    def forward_pass(self, x):

        ft_mode = (len(x) == 2)
        if ft_mode:
            x_state, x_feed = x
        else:
            x_state = x
        z_enc, z_latent = self.encoder(x_state)

        decoder_input = [z_enc, x_feed] if ft_mode else z_enc
        y_pred = self.decoder(decoder_input)

        return y_pred, z_latent

    def train_step_AE(self, data):
        x, y = data
        y_pred, z = self.forward_pass(x)

        # time ordering in y is backwards so last first
        y_true = y[0][:, 0,]

        # create a mask on the fly ## FIXME: factorize
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        logical_mask = ~ops.isnan(y_true)
        y_true = y_true[logical_mask]
        y_pred = y_pred[logical_mask]

        total_loss = self.loss_fn(y_true, y_pred)
        total_loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        self.total_loss_tracker.update_state(total_loss)
        out_dict = {
            'loss': self.total_loss_tracker.result(),
        }

        return out_dict

    def train_step_RNN(self, data):
        x, y = data
        y_pred, z = self.forward_pass(x)

        _, z_true = self.encoder(y)
        rnn_loss = self.loss_fn(z_true['rnn_input'],
                                z['rnn_output'])

        # ignore nan result
        rnn_loss = 0 if ops.isnan(rnn_loss) else rnn_loss

        # time ordering in y is backwards so last first
        y_true = y[0][:, 0,]

        # create a mask on the fly ## FIXME: factorize
        #   ## FIXME FIXME
        crop = 10
        y_true = y_true[:, crop:-crop, crop:-crop, :]
        y_pred = y_pred[:, crop:-crop, crop:-crop, :]
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        logical_mask = ~ops.isnan(y_true)
        y_true = y_true[logical_mask]
        y_pred = y_pred[logical_mask]

        reconstruction_loss = self.loss_fn(y_true, y_pred)

        total_loss = reconstruction_loss + rnn_loss

        total_loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.rnn_loss_tracker.update_state(rnn_loss)

        out_dict = {
            'loss': self.total_loss_tracker.result(),
            'reconstr_loss': self.reconstruction_loss_tracker.result(),
            'RNN_loss': self.rnn_loss_tracker.result(),
        }

        return out_dict

    def train_step_VAE(self, data, RNN_hybrid=False):
        x, y = data
        y_pred, z = self.forward_pass(x)

        if RNN_hybrid:
            _, z_true = self.encoder(y)

            rnn_loss_mean = \
                ops.mean(
                    ops.sum(
                        ops.square(z_true['mean'] - z['rnn_mean']),
                        axis=(1, 2),
                    )
                )

            rnn_loss_var = \
                ops.mean(
                    ops.sum(
                        ops.square(z_true['log_var'] - z['rnn_log_var']),
                        axis=(1, 2),
                    )
                )

            rnn_loss = rnn_loss_mean + rnn_loss_var
            # ignore nan result
            rnn_loss = 0 if ops.isnan(rnn_loss) else rnn_loss

            self.rnn_loss_tracker.update_state(rnn_loss)
            rnn_dict = {'rnn_loss': self.rnn_loss_tracker.result()}

        # time ordering in y is backwards so last first
        y_true = y[0][:, 0,]

        # create a mask on the fly ## FIXME: factorize
        
        crop = 10
        y_true = y_true[:, crop:-crop, crop:-crop, :]
        y_pred = y_pred[:, crop:-crop, crop:-crop, :]

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        logical_mask = ~ops.isnan(y_true)
        y_true = y_true[logical_mask]
        y_pred = y_pred[logical_mask]

        #  ## FIXME the loss can be artifically high because of the
        #  ## boundaries. Crop the error.  !! Factorize the masked
        #  ## loss first 

        # time ordering in y is backwards so last first
        reconstruction_loss = \
            ops.mean(ops.square(y_true - y_pred))

        kl_loss = \
            -0.5 * (1 + z['log_var'] - ops.square(z['mean']) -
                    ops.exp(z['log_var']))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=(1, 2)))

        if RNN_hybrid:
            total_loss = reconstruction_loss + kl_loss + rnn_loss
        else:
            total_loss = reconstruction_loss + kl_loss

        total_loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        out_dict = {
            'loss': self.total_loss_tracker.result(),
            'reconstr_loss': self.reconstruction_loss_tracker.result(),
            'KL_loss': self.kl_loss_tracker.result()
        }
        if RNN_hybrid:
            out_dict.update(rnn_dict)

        return out_dict


@keras.saving.register_keras_serializable(name="sampling")
class Sampling(layers.Layer):
    """
    Sampling layer
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # optional seed here

    def call(self, mean, log_var):
        eps = keras.random.normal(
            shape=ops.shape(mean)
        )
        out = mean + ops.exp(0.5 * log_var) * eps
        return out

    def get_config(self):
        config = super().get_config()
        return config



# custom masking class
@keras.saving.register_keras_serializable(name="custom_masking")
class Masking(layers.Layer):
    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        self.mask = mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'mask': keras.saving.serialize_keras_object(self.mask)})
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
            losstype='MSE'
    ):
        super().__init__(name=name,
                         reduction=reduction)
        self.losstype = losstype

    def call(self, y_true, y_pred):

        if self.losstype == 'NSE':
            loss = self.normalized_SE(y_true, y_pred)
        elif self.losstype == 'MSE':
            loss = self.mean_SE(y_true, y_pred)

        print(f':{loss:1.2e}: ', end="")
        return loss

    def normalized_SE(self, y_true, y_pred):
        err = ops.sum(ops.square(y_pred - y_true))
        nrm = ops.sum(ops.square(y_true))
        return (err / nrm)

    def mean_SE(self, y_true, y_pred):
        loss = ops.mean(ops.square(y_pred - y_true))
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
                 pars,
                 scalers=None,
                 case_study='cmems'
                 ):

        super().__init__()

        self.pars = pars
        self.scalers = scalers
        self.unroll_dim = self.pars['unroll_dim']
        self.data = data
        self.test_inds = test_inds
        self.case_study = case_study

        # account for unrolling
        max_inds = self.data['HR'].shape[0]
        unr_inds = np.arange(test_inds[-1] + 1,
                             np.min([test_inds[-1] + self.unroll_dim + 1,
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
        if self.case_study == 'cmems':
            self.predict_cmems(epoch, logs)
        elif self.case_study == 'swot':
            self.predict_swot(epoch, logs)
        else:
            raise ValueError("invalid case study")

    def predict_swot(self, epoch, logs=None):
        self.predictions = np.zeros_like(self.test_data)

        pb_i = keras.utils.Progbar(self.N_steps,
                                   stateful_metrics=['error', 'base'],
                                   interval=0.5)

        if self.unroll_dim != 0:
            raise NotImplementedError("unroll not implemented for SWOT set")

        def crop(field):
            return field[:, 10:-10, 10:-10, :]

        error, base = (0, 0)
        for i in range(self.N_steps):
            xk_lb = np.expand_dims(
                data_utils.create_lookback(
                    self.test_inds[i], [self.data['LR']],
                    self.lookback, axis=0)[0], axis=0)

            # get rid of truth
            xk_lb = np.stack([
                xk_lb[:, :, :, :, 0],
                xk_lb[:, :, :, :, 0]
            ], axis=-1)
            xk = self.model.predict([xk_lb, xk_lb], verbose=0)

            self.predictions[i,] = xk
            if self.pars['evaluate']:
                # breakpoint()
                xk_true = np.expand_dims(self.test_data[i,], axis=0)

                error += (np.nansum(np.square(crop(xk - xk_true))))

                xk_ref = np.expand_dims(
                    self.data['LR'][self.test_inds[i],], axis=0)

                base += (np.nansum(np.square(crop(xk_ref - xk_true))))

                values = [('error', np.sqrt(error / (i + 1))),
                          ('base', np.sqrt(base / (i + 1)))]
                pb_i.add(1, values=values)
            else:
                pb_i.add(1)

        import matplotlib.pyplot as plt
        plt.close('all')

        plt.figure(figsize=(10, 10))

        t, x, y, nc = xk.shape
        xk_unscaled = self.scalers['LR']\
                          .inverse_transform(xk.reshape(t, -1))\
                          .reshape(t, x, y, nc)

        # xk_unscaled = xk

        t, x, y, nc = xk_ref.shape
        xk_ref_unscaled = self.scalers['LR']\
                              .inverse_transform(xk_ref.reshape(t, -1))\
                              .reshape(t, x, y, nc)
        # xk_ref_unscaled = xk_ref

        plt.subplot(2, 2, 1)
        c = plt.imshow(xk_ref_unscaled[0, :, :, 0])
        plt.colorbar(c)
        plt.title('coarse')

        plt.subplot(2, 2, 2)
        c = plt.imshow(xk_unscaled[0, :, :, 0])
        plt.colorbar(c)
        plt.title('corrected')

        errorfield = np.square(crop(xk - xk_true))
        print(np.nanmax(error))

        plt.subplot(2, 2, 3)
        c = plt.imshow(xk_true[0, :, :, 0])
        plt.colorbar(c)
        plt.subplot(2, 2, 4)
        c = plt.imshow(errorfield[0, :, :, 0])
        plt.title('error')
        plt.colorbar(c)
        plt.pause(1)
        if self.pars['evaluate']:
            # self.plotmachine.plot_prediction_error(self.test_data,
            #                                        self.predictions,
            #                                        self.test_data_ft,
            #                                        f'epoch_{epoch}')

            self.final_error = np.sqrt(error / (i + 1))
            self.final_base = np.sqrt(base / (i + 1))
            logs['error'] = self.final_error
            logs['base']  = self.final_base


    def predict_cmems(self, epoch, logs=None):
        self.predictions = np.zeros_like(self.test_data)

        init_ind = self.test_inds[0] - 1

        xk_lb = np.expand_dims(
            data_utils.create_lookback(init_ind, [self.data['HR']],
                                       self.lookback, axis=0)[0], axis=0)

        pb_i = keras.utils.Progbar(self.N_steps,
                                   stateful_metrics=['error', 'base'],
                                   interval=0.5)
        error, base = (0, 0)

        for i in range(self.N_steps - self.unroll_dim):
            xk_LR = [np.expand_dims(
                data_utils.create_lookback(self.test_inds[i + unroll],
                                           [self.data['LR']],
                                           self.lookback, axis=0)[0], axis=0)
                     for unroll in range(self.unroll_dim + 1)
                     ]

            Pxk = xk_LR

            if self.pars['feedthrough_only']:
                xk = self.model.predict([Pxk[0]], verbose=0)
            elif self.pars['use_feedthrough']:
                xk = self.model.predict([xk_lb] + Pxk, verbose=0)
            else:
                xk = self.model.predict([xk_lb], verbose=0)

            if (
                    isinstance(xk, list) and
                    self.unroll_dim > 0
            ):
                for j in range(self.unroll_dim + 1):
                    self.predictions[i + j,] = xk[j]
                xk = xk[0]
            else:
                self.predictions[i,] = xk

            if self.pars['evaluate']:
                xk_true = np.expand_dims(self.test_data[i,], axis=0)
                error += (np.sum(np.square(xk - xk_true)))
                base += (np.sum(np.square(xk_LR[0][:, 0,] - xk_true)))
                values = [('error', np.sqrt(error / (i + 1))),
                          ('base', np.sqrt(base / (i + 1)))]

                pb_i.add(1, values=values)
            else:
                pb_i.add(1)

            xk = np.expand_dims(xk, axis=1)
            xk_lb = np.concatenate(
                [xk, xk_lb], axis=1
            )[:, :self.lookback + 1,]

        if self.pars['evaluate']:
            self.plotmachine.plot_prediction_error(self.test_data,
                                                   self.predictions,
                                                   self.test_data_ft,
                                                   f'epoch_{epoch}')

            self.final_error = np.sqrt(error / (i + 1))
            self.final_base = np.sqrt(base / (i + 1))
            logs['error'] = self.final_error
            logs['base']  = self.final_base
        os.environ["DISPLAY"] = ":0"
