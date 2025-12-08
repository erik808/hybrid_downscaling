import torch
import keras
from keras import ops
from keras import layers
from keras import regularizers

import numpy as np
import importlib
import tools
import base_model
import resnet_model
import vae_model

importlib.reload(base_model)
importlib.reload(resnet_model)
importlib.reload(vae_model)


class Predictor(base_model.BaseModel):
    def __init__(
            self,
            vae_model,
            **kwargs,
    ):
        super().__init__(**kwargs)

        tools.load_config(self, config_name='predictor_model')

        # get input and output layers to isolate encoder+decoder
        mean, lvar =\
            vae_model.model\
                     .get_layer('vae_splitter').output

        sampled = \
            vae_model.model\
                     .get_layer('vae_sampling').output
        encoder_input = \
            vae_model.model\
                     .get_layer('vae_input_transform').input
        decoder_output = \
            vae_model.model\
                     .get_layer('vae_masking').output

        decoder_skip_output = \
            vae_model.model\
                     .get_layer('vae_hybrid_coupling').output

        self.vae_input = vae_model.model.input['HR_data']

        self.encoder = keras.Model(
            inputs=encoder_input,
            outputs=[mean, lvar],
            name="encoder",
        )
        self.encoder.build(encoder_input.shape)

        self.sampler = vae_model.model.get_layer('vae_sampling')

        self.hidden_state_dim = \
            self.esn_dmd_pars['Nr'] if 'ESN' in self.predictor else \
            self.data_gen.dm.hidden_states.shape[1]

        hidden_shape = (
            self.data_gen.dm.hidden_states.shape[0],
            self.hidden_state_dim,
        )
        self.data_gen.dm.hidden_states = np.zeros(hidden_shape)

        self.decoder = keras.Model(
            inputs=sampled,
            outputs=[decoder_output, decoder_skip_output],
            name="decoder",
        )
        self.decoder.build(sampled.shape)

        self.encoder.trainable = self.trainable_encoder
        self.decoder.trainable = self.trainable_decoder
        self.trainable_VAE = \
            self.trainable_encoder or self.trainable_decoder

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        self.trackers = []
        self.trackers.append(keras.metrics.Mean(name="loss"))

        for loss_name in self.loss_list:
            self.trackers.append(keras.metrics.Mean(name=loss_name))

        self.loss_KL = vae_model.loss_KL

    @property
    def metrics(self):
        return self.trackers

    def create_input(self, inputs):
        return {
            self.input_name_HR:
            ops.nan_to_num(inputs[self.input_name_HR]),
            self.input_name_LR:
            ops.nan_to_num(inputs[self.input_name_LR]),
            'hidden':
            inputs['hidden'],
        }

    def train_step(self, data, training=True):
        x, y = data

        if training:
            self.zero_grad()

        z = self(x, training=training)

        z_decoded = z['decoded'][:,
                                 self.masking.rows,
                                 self.masking.cols,
                                 :]

        if 'KL' in self.loss_list:
            z_mean = z['mean']
            z_logvar = z['logvar']
            # kl loss variance formulation
            kl_loss = self.loss_KL(z_mean, z_logvar, beta=self.beta)
        else:
            kl_loss = 0.0

        if 'inner_pred' in self.loss_list:
            z_ls_pred = z['ls_pred']
            y_ls = \
                self.encoder(
                    ops.nan_to_num(
                        ops.squeeze(
                            y[self.input_name_HR][:,
                                                  0,  # target lookback index
                                                  ...],
                            axis=1)),
                    training=training,
                )[0]  # take only the mean

            # prediction loss in the latent space
            lspred_loss = self.loss_MSE(z_ls_pred, y_ls) * self.alpha_inner
        else:
            lspred_loss = 0.0

        def y_k(k):
            return \
                y[self.input_name_HR][:,
                                      k,  # kth lookback index
                                      self.masking.rows,
                                      self.masking.cols,
                                      :]

        if 'reconstruction' in self.loss_list:
            # reconstruction loss, compare using most recent
            z_ae_recons = z['ae_recons'][:,
                                         self.masking.rows,
                                         self.masking.cols,
                                         :]
            re_loss = self.loss_MSLE(z_ae_recons, y_k(1)) * self.gamma
        else:
            re_loss = 0.0

        if 'outer_pred' in self.loss_list:
            # prediction loss, compare against target
            pred_loss = self.loss_MSLE(z_decoded, y_k(0)) * self.alpha_outer
        else:
            pred_loss = 0.0

        if 'ls_size' in self.loss_list:
            # latent space size loss
            z_mean = z['mean']
            ls_size = ops.mean(ops.abs(z_mean)) * self.alpha_ls
        else:
            ls_size = 0.0

        # combine losses
        loss = pred_loss + lspred_loss + re_loss + kl_loss + ls_size

        if training:
            loss.backward()
            trainable_weights = [v for v in self.trainable_weights]
            gradients = [v.value.grad for v in trainable_weights]

            # update weights
            with torch.no_grad():
                self.optimizer.apply(gradients, trainable_weights)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            if metric.name == "outer_pred":
                metric.update_state(pred_loss)
            if metric.name == "inner_pred":
                metric.update_state(lspred_loss)
            if metric.name == "reconstruction":
                metric.update_state(re_loss)
            if metric.name == "ls_size":
                metric.update_state(ls_size)
            if metric.name == "KL":
                metric.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def builder(self):

        # reusing the vae input layer
        input_HR = self.vae_input

        # check dimensions
        _, lbdim, _, _, _ = input_HR.shape
        assert lbdim > 1, "need at least lookback=2 to make predictions"

        timeseries = ops.split(
            input_HR,
            self.input_shape_HR[0],
            axis=1)

        # lookback ordering is backwards in time, reversing to get it
        # forwards in time
        timeseries.reverse()

        # remove current lookback (our target), keep only past samples
        timeseries.pop()

        # encode timeseries using encoder
        encoded_series = [self.encoder(ops.squeeze(snapshot,
                                                   axis=1))
                          for snapshot in timeseries]

        # use most recent lookback for reconstruction loss
        mean, logvar = encoded_series[-1]
        sampled = self.sampler(mean, logvar)
        ae_reconstruction, _ = self.decoder(sampled)

        # continue with only the mean
        encoded_series = [sample[0] for sample in encoded_series]

        # get control input
        input_LR = layers.Input(
            shape=self.input_shape_LR,
            name='LR_data')
        control = ops.split(
            input_LR,
            self.input_shape_LR[0],
            axis=1)[0]

        # get hidden state
        hidden_state = layers.Input(
            shape=(self.hidden_state_dim,),
            name='hidden_state',
        )

        # do prediction in latent space
        prediction, hidden_state_new = LSPredictor(
            name="latent_predictor"
        )(
            encoded_series,
            hidden_state,
            control,
        )

        prediction_decoded, skipped = self.decoder(prediction)

        outputs = {
            'decoded': prediction_decoded,
            'ls_pred': prediction,
            'mean': mean,
            'hidden': hidden_state_new,
            'logvar': logvar,
            'ae_recons': ae_reconstruction,
            'skip_vae_output': skipped,
        }
        inputs = {
            self.input_name_HR: input_HR,
            self.input_name_LR: input_LR,
            'hidden': hidden_state,
        }
        return inputs, outputs


class LSPredictor(layers.Layer):
    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)
        tools.load_config(self, config_name='predictor_model')

        if self.kernel_regularizer is not None:
            key, value = next(iter(self.kernel_regularizer.items()))
            self.kernel_regularizer = getattr(regularizers, key)(value)

        if self.recurrent_regularizer is not None:
            key, value = next(iter(self.recurrent_regularizer.items()))
            self.recurrent_regularizer = getattr(regularizers, key)(value)

    def build(self, input_shape):
        dims = input_shape[0][1:]  # ignore batch dim
        lb_dim = len(input_shape)
        self.combine_inputs = CombineInputs(mode='bypass')

        if self.predictor == 'simpleRNN':
            self.input_transf = FlattenAndStack()
            self.predictmod = layers.SimpleRNN(
                units=self.dense_units,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=self.kernel_regularizer,
                recurrent_regularizer=self.recurrent_regularizer,
                unroll=False
            )
            self.output_transf = \
                keras.Sequential([
                    layers.Dense(units=np.prod(dims),
                                 kernel_regularizer=self.kernel_regularizer,
                                 activation=self.activation),
                    layers.Reshape(dims),
                ])

        elif self.predictor == 'lstm':
            self.input_transf = FlattenAndStack()
            self.predictmod = layers.LSTM(
                units=self.dense_units,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=self.kernel_regularizer,
                recurrent_regularizer=self.recurrent_regularizer,
                unroll=False
            )
            self.output_transf = \
                keras.Sequential([
                    layers.Dense(units=np.prod(dims),
                                 kernel_regularizer=self.kernel_regularizer,
                                 activation=self.activation),
                    layers.Reshape(dims),
                ])

        elif self.predictor == 'dense':
            self.input_transf = keras.Sequential([
                FlattenAndStack(),
                layers.Flatten(),
            ])
            self.predictmod = keras.Sequential([
                layers.Dense(
                    units=self.dense_units,
                    activation=self.activation,
                ),
                layers.Dense(
                    units=np.prod(dims),
                    activation=self.activation,
                )
            ])
            self.output_transf = layers.Reshape(dims)

        elif self.predictor == 'conv3d':
            self.input_transf = Stack()

            self.predictmod = keras.Sequential([
                layers.Conv3D(
                    filters=256,
                    kernel_size=(np.ceil(lb_dim / 2), 3, 3),
                    strides=1,
                    padding='same',
                    activation=self.activation,
                    kernel_regularizer=self.kernel_regularizer,
                ),
                layers.Conv3D(
                    filters=128,
                    kernel_size=(lb_dim, 3, 3),
                    strides=1,
                    padding='same',
                    activation=self.activation,
                    kernel_regularizer=self.kernel_regularizer,
                ),
                layers.Conv3D(
                    filters=self.output_filters,
                    kernel_size=(lb_dim, 3, 3),
                    strides=1,
                    padding='same',
                    activation=self.activation,
                    kernel_regularizer=self.kernel_regularizer,
                ),
            ])

            self.output_transf = keras.Sequential([
                layers.MaxPooling3D(
                    pool_size=(lb_dim, 1, 1),
                    padding='same'),
                Squeeze(),
            ])

        elif self.predictor == 'convlstm':
            self.input_transf = Stack()
            self.predictmod = layers.ConvLSTM2D(
                filters=self.convlstm_filters,
                kernel_size=3,
                strides=1,
                padding='same',
            )
            self.output_transf = layers.Conv2D(
                filters=self.output_filters,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=self.activation)

        elif self.predictor == 'identity':
            self.input_transf = Last()
            self.predictmod = layers.Identity()
            self.output_transf = layers.Identity()

        elif (
                self.predictor == 'DMD' or
                self.predictor == 'DMDc' or
                self.predictor == 'ESN' or
                self.predictor == 'ESNc'
        ):
            self.input_transf = keras.Sequential([
                Last(),
                layers.Flatten(),
            ])
            self.combine_inputs = CombineInputs(mode='append')
            self.predictmod = DMD(
                name='dmd_operator',
                mode=self.predictor,
                alpha=self.esn_dmd_pars['alpha'],
                use_bias=self.esn_dmd_bias,
                squaredStates=self.esn_dmd_pars['squaredStates'],
            )
            self.output_transf = layers.Reshape(dims)
        else:
            raise Exception("Invalid predictor")

    def call(self, inputs, hidden=None, control=None):
        x = self.input_transf(inputs)
        x = self.combine_inputs(x, hidden, control)
        x = self.predictmod(x)

        if isinstance(x, tuple):
            x, hidden = x

        out = self.output_transf(x)
        # print(hidden[0, :5])
        return out, hidden


class Stack(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return ops.stack(x, axis=1)


class Last(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x[-1]


class Squeeze(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return ops.squeeze(x, axis=1)


class FlattenAndStack(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return ops.stack([layers.Flatten()(sample)
                          for sample in x], axis=1)


class DMD(layers.Layer):
    def __init__(
            self,
            mode='DMD',
            alpha=1.0,
            use_bias=False,
            squaredStates='even',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.alpha = alpha
        self.use_bias = use_bias
        self.squaredStates = squaredStates
        self.concat = False

    def build(self, input_shape):

        if isinstance(input_shape[0], tuple):
            xk_shape, hidden_shape, control_shape = input_shape
        else:
            raise Exception('DMD needs to be called '
                            'with hidden and control inputs')

        W_shape = (hidden_shape[-1], hidden_shape[-1])
        self.Nr = W_shape[0]

        if self.mode == 'DMD':
            self.concat = False
            W_in_shape = (
                hidden_shape[-1],
                xk_shape[-1],
            )
            W_out_shape = (
                xk_shape[-1],
                xk_shape[-1],
            )
            bias_shape = (W_out_shape[0],)

        elif self.mode == 'DMDc':
            self.concat = True
            W_in_shape = (
                hidden_shape[-1],
                xk_shape[-1] + control_shape[-1],
            )
            W_out_shape = (
                xk_shape[-1],
                xk_shape[-1] + control_shape[-1],
            )
            bias_shape = (W_out_shape[0],)

        elif self.mode == 'ESN':
            W_in_shape = (
                hidden_shape[-1],
                xk_shape[-1],
            )
            W_out_shape = (
                xk_shape[-1],
                hidden_shape[-1],
            )
            bias_shape = (hidden_shape[-1],)

        elif self.mode == 'ESNc':
            W_in_shape = (
                hidden_shape[-1],
                xk_shape[-1] + control_shape[-1],
            )
            W_out_shape = (
                xk_shape[-1],
                hidden_shape[-1] + control_shape[-1],
            )
            bias_shape = (hidden_shape[-1],)

        self.W = self.add_weight(
            shape=W_shape,
            trainable=False,
            name='W',
        )
        self.W_in = self.add_weight(
            shape=W_in_shape,
            trainable=False,
            name='W_in',
        )
        self.W_out = self.add_weight(
            shape=W_out_shape,
            trainable=False,
            name='W_out',
        )

        self.bias = self.add_weight(
            shape=bias_shape,
            initializer='zeros',
            trainable=True,
            name='DMD_bias',
        )

        self.built = True

    def call(self, inputs):
        if 'DMD' in self.mode:
            return self.call_DMD(inputs)
        elif 'ESN' in self.mode:
            return self.call_ESN(inputs)

    def call_ESN(self, inputs):
        """this is basically a keras of equivalent of the implementation in
ESN"""

        xkm1, hidden, control = inputs

        # update hidden state
        u_in = ops.concatenate([xkm1, control], -1)\
            if self.mode == 'ESNc' else xkm1

        pre = (ops.matmul(self.W, hidden.T).T +
               ops.matmul(self.W_in, u_in.T).T)

        if self.use_bias:
            pre = ops.add(pre, self.bias)

        hidden = self.alpha * ops.tanh(pre) + (1 - self.alpha) * hidden

        # copy tensor
        hidden_pre = ops.add(hidden, 0)

        # apply squaredStates
        if self.squaredStates == 'even':
            even_inds = range(1, self.Nr, 2)
            hidden_pre[..., even_inds] = hidden_pre[..., even_inds]**2

        # compute prediction
        if self.mode == 'ESNc':
            x = ops.concatenate([control, hidden_pre], -1)
        else:
            x = hidden_pre

        output = ops.matmul(self.W_out, x.T).T

        return output, hidden

    def call_DMD(self, inputs):
        if self.concat and isinstance(inputs, list):
            # ignore the hidden state
            inputs = ops.concatenate([inputs[0], inputs[2]], -1)
        elif isinstance(inputs, list):
            inputs = inputs[0]

        output = ops.matmul(self.W_out, inputs.T).T
        if self.use_bias:
            output = ops.add(output, self.bias)

        return output


class CombineInputs(layers.Layer):
    def __init__(
            self,
            mode='bypass',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode

    def build(self, input_shape):
        self.flat = layers.Flatten()
        self.built = True

    def call(self, inputs, hidden=None, control=None):

        x = inputs
        if hidden is not None:
            x = [x, hidden]

        if control is not None:
            control = self.flat(control)
            x = [*x, control] if isinstance(x, list) else [x, control]

        if self.mode == 'append':
            return x
        elif self.mode == 'bypass':
            return inputs
        else:
            raise Exception('invalid mode')
