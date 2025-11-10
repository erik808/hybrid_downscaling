import torch
import keras
from keras import ops
from keras import layers

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

        self.vae_model = vae_model

        # get input and output layers to isolate encoder+decoder
        mean, logsigma = \
            self.vae_model.get_layer('betaVAE')\
                          .get_layer('vae_splitter').output
        sampled = \
            self.vae_model.get_layer('betaVAE')\
                          .get_layer('vae_sampling').output
        vae_input = \
            self.vae_model.get_layer('betaVAE')\
                          .get_layer('vae_input_transform').input
        vae_output = \
            self.vae_model.get_layer('betaVAE')\
                          .get_layer('vae_masking').output

        self.encoder = keras.Model(
            inputs=vae_input,
            outputs=mean,
            name="encoder",
        )

        self.decoder = keras.Model(
            inputs=sampled,
            outputs=vae_output,
            name="decoder",
        )

        self.encoder.trainable = self.trainable_encoder
        self.decoder.trainable = self.trainable_decoder

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.pred_loss_tracker = keras.metrics.Mean(name="prediction")
        self.lspred_loss_tracker = keras.metrics.Mean(name="ls_pred")
        self.re_loss_tracker = keras.metrics.Mean(name="reconstruction")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.pred_loss_tracker,
            self.lspred_loss_tracker,
            self.re_loss_tracker,
        ]

    def train_step(self, data, training=True):
        x, y = data
        if training:
            self.zero_grad()

        z = self({'HR_data': ops.nan_to_num(x['HR_data'])},
                 training=training)

        z_decoded = z['decoded'][:,
                                 self.masking.rows,
                                 self.masking.cols,
                                 :]
        z_ls_pred = z['ls_pred']
        z_ae_proj = z['ae_proj'][:,
                                 self.masking.rows,
                                 self.masking.cols,
                                 :]

        y_ls = \
            self.encoder(
                ops.nan_to_num(
                    ops.squeeze(y['HR_data'][:,
                                             0,  # target lookback index
                                             ...],
                                axis=1)))

        # prediction loss in the latent space
        lspred_loss = self.loss_fn(z_ls_pred, y_ls)

        def y_k(k):
            return \
                y['HR_data'][:,
                             k,  # kth lookback index
                             self.masking.rows,
                             self.masking.cols,
                             :]

        # prediction loss, compare against target
        pred_loss = self.loss_fn(z_decoded, y_k(0))

        # reconstruction loss, compare using most recent, only used
        # when the VAE weights are trainable
        if self.trainable_VAE:
            re_loss = self.loss_fn(z_ae_proj, y_k(1))
        else:
            re_loss = 0.0

        # combine losses
        loss = pred_loss + lspred_loss + re_loss

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

        return {m.name: m.result() for m in self.metrics}

    def builder(self):
        inputs = layers.Input(
            shape=self.input_shape_HR,
            name=self.input_name_HR)

        # check dimensions
        _, lbdim, _, _, _ = inputs.shape
        assert lbdim > 1, "need at least lookback=2 to make predictions"

        timeseries = ops.split(
            inputs,
            self.input_shape_LR[0],
            axis=1)

        # lookback ordering is backwards in time, reversing to get it
        # forwards in time
        timeseries.reverse()

        # remove current lookback (our target), keep only past samples
        timeseries.pop()

        # use most recent lookback for reconstruction loss
        if self.trainable_VAE:
            ae_projection = \
                self.decoder(
                    self.encoder(ops.squeeze(timeseries[-1],
                                             axis=1)))
        else:
            ae_projection = ops.squeeze(timeseries[-1],
                                        axis=1)

        # encode timeseries
        encoded_series = [self.encoder(ops.squeeze(sample,
                                                   axis=1))
                          for sample in timeseries]

        # do prediction
        prediction = LSPredictor(self.predictor)(encoded_series)

        prediction_decoded = self.decoder(prediction)
        outputs = {
            'decoded': prediction_decoded,
            'ls_pred': prediction,
            'ae_proj': ae_projection,
        }

        return inputs, outputs


class LSPredictor(layers.Layer):
    def __init__(
            self,
            mode,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode

    def build(self, input_shape):
        dims = input_shape[0][1:]  # ignore batch dim
        lb_dim = len(input_shape)

        if self.mode == 'simpleRNN':
            self.input_transf = FlattenAndStack()
            self.model = layers.SimpleRNN(
                units=128,
                recurrent_dropout=0.4,
                unroll=False
            )
            self.output_transf = \
                keras.Sequential([
                    layers.Dense(units=np.prod(dims),
                                 activation='leaky_relu'),
                    layers.Reshape(dims),
                ])

        elif self.mode == 'dense':
            self.input_transf = keras.Sequential([
                FlattenAndStack(),
                layers.Flatten(),
            ])
            self.model = keras.Sequential([
                layers.Dense(
                    units=128,
                    activation='leaky_relu',
                ),
                layers.Dense(
                    units=np.prod(dims),
                    activation='leaky_relu',
                )
            ])
            self.output_transf = layers.Reshape(dims)

        elif self.mode == 'conv3d':
            self.input_transf = Stack()

            self.model = keras.Sequential([
                layers.Conv3D(
                    filters=256,
                    kernel_size=(int(lb_dim / 2), 3, 3),
                    strides=1,
                    padding='same',
                    activation='leaky_relu',
                ),
                layers.Conv3D(
                    filters=128,
                    kernel_size=(lb_dim, 2, 2),
                    strides=1,
                    padding='same',
                    activation='leaky_relu',
                ),
                layers.Conv3D(
                    filters=dims[-1],
                    kernel_size=(lb_dim, 2, 2),
                    strides=1,
                    padding='same',
                    activation='leaky_relu',
                ),
            ])

            self.output_transf = keras.Sequential([
                layers.MaxPooling3D(
                    pool_size=(lb_dim, 1, 1),
                    padding='same'),
                Squeeze(),
            ])

        elif self.mode == 'convlstm':
            self.input_transf = Stack()
            self.model = layers.ConvLSTM2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
            )
            self.output_transf = layers.Identity()

    def call(self, inputs):
        x = self.input_transf(inputs)
        x = self.model(x)
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
