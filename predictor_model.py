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
                     .get_layer('skip_output').output

        self.vae_input = vae_model.model.input['HR_data']

        self.encoder = keras.Model(
            inputs=encoder_input,
            outputs=[mean, lvar],
            name="encoder",
        )
        self.encoder.build(encoder_input.shape)

        self.sampler = vae_model.model.get_layer('vae_sampling')

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

        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_KL = vae_model.loss_KL
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.re_loss_tracker = keras.metrics.Mean(name="reconstruction")
        self.KL_loss_tracker = keras.metrics.Mean(name="KLloss")
        self.pred_loss_tracker = keras.metrics.Mean(name="prediction")
        self.lspred_loss_tracker = keras.metrics.Mean(name="ls_pred")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.re_loss_tracker,
            self.KL_loss_tracker,
            self.pred_loss_tracker,
            self.lspred_loss_tracker,
        ]

    def create_input(self, inputs):
        return {self.input_name_HR:
                ops.nan_to_num(inputs[self.input_name_HR])}

    def train_step(self, data, training=True):
        x, y = data
        if training:
            self.zero_grad()

        z = self(x, training=training)

        z_decoded = z['decoded'][:,
                                 self.masking.rows,
                                 self.masking.cols,
                                 :]
        z_ls_pred = z['ls_pred']
        z_ae_recons = z['ae_recons'][:,
                                     self.masking.rows,
                                     self.masking.cols,
                                     :]
        z_mean = z['mean']
        z_logvar = z['logvar']
        # kl loss variance formulation
        kl_loss = self.loss_KL(z_mean, z_logvar, beta=self.beta)

        y_ls = \
            self.encoder(
                ops.nan_to_num(
                    ops.squeeze(
                        y[self.input_name_HR][:,
                                              0,  # target lookback index
                                              ...],
                        axis=1)))[0]  # take only the mean

        # prediction loss in the latent space
        lspred_loss = self.loss_fn(z_ls_pred, y_ls)

        def y_k(k):
            return \
                y[self.input_name_HR][:,
                                      k,  # kth lookback index
                                      self.masking.rows,
                                      self.masking.cols,
                                      :]

        # prediction loss, compare against target
        pred_loss = self.loss_fn(z_decoded, y_k(0))

        # reconstruction loss, compare using most recent, only used
        # when the VAE weights are trainable
        if self.trainable_VAE:
            re_loss = self.loss_fn(z_ae_recons, y_k(1)) * self.gamma
        else:
            re_loss = 0.0

        # combine losses
        loss = pred_loss + lspred_loss + re_loss + kl_loss

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
            if metric.name == "prediction":
                metric.update_state(pred_loss)
            if metric.name == "ls_pred":
                metric.update_state(lspred_loss)
            if metric.name == "reconstruction":
                metric.update_state(re_loss)
            if metric.name == "KLloss":
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
            self.input_shape_LR[0],
            axis=1)

        # lookback ordering is backwards in time, reversing to get it
        # forwards in time
        timeseries.reverse()

        # remove current lookback (our target), keep only past samples
        timeseries.pop()

        # use most recent lookback for reconstruction loss
        mean, logvar = self.encoder(
            ops.squeeze(timeseries[-1], axis=1))
        sampled = self.sampler(mean, logvar)
        ae_reconstruction, _ = self.decoder(sampled)

        # encode timeseries using mean output of encoder
        encoded_series = [self.encoder(ops.squeeze(snapshot,
                                                   axis=1))[0]
                          for snapshot in timeseries]

        # do prediction in latent space
        prediction = LSPredictor(
            name="latent_predictor",
        )(encoded_series)

        prediction_decoded, skipped = self.decoder(prediction)
        outputs = {
            'decoded': prediction_decoded,
            'ls_pred': prediction,
            'mean': mean,
            'logvar': logvar,
            'ae_recons': ae_reconstruction,
            'skip_vae_output': skipped,
        }
        inputs = {self.input_name_HR: input_HR}
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
                    kernel_size=(lb_dim, 2, 2),
                    strides=1,
                    padding='same',
                    activation=self.activation,
                    kernel_regularizer=self.kernel_regularizer,
                ),
                layers.Conv3D(
                    filters=dims[-1],
                    kernel_size=(lb_dim, 2, 2),
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
                filters=dims[-1],
                kernel_size=3,
                strides=1,
                padding='same',
                activation=self.activation)

        else:
            raise Exception("Invalid predictor")

    def call(self, inputs):
        x = self.input_transf(inputs)
        x = self.predictmod(x)
        out = self.output_transf(x)
        return out


class Stack(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return ops.stack(x, axis=1)


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
