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


class RNN(base_model.BaseModel):
    def __init__(
            self,
            vae_model,
            **kwargs,
    ):
        super().__init__(**kwargs)

        tools.load_config(self, config_name='rnn_model')

        self.vae_model = vae_model

        # get input and output layers to isolate encoder+decoder
        mean, logsigma = \
            self.vae_model.get_layer('betaVAE')\
                          .get_layer('splitter').output
        sampled = \
            self.vae_model.get_layer('betaVAE')\
                          .get_layer('sampling').output
        vae_input = \
            self.vae_model.get_layer('betaVAE')\
                          .get_layer('input_transform').input
        vae_output = \
            self.vae_model.get_layer('betaVAE')\
                          .get_layer('masking').output

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

        if not self.trainable_VAE:
            # disable training on VAE model
            self.vae_model.trainable = False
            self.encoder.trainable = False
            self.decoder.trainable = False

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.pred_loss_tracker = keras.metrics.Mean(name="prediction")
        self.rnn_loss_tracker = keras.metrics.Mean(name="rnn")
        self.re_loss_tracker = keras.metrics.Mean(name="reconstruction")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.pred_loss_tracker,
            self.rnn_loss_tracker,
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
                                             0,  # current lookback index
                                             ...])))

        rnn_loss = self.loss_fn(z_ls_pred, y_ls)

        y = y['HR_data'][:,
                         0,  # current lookback index
                         self.masking.rows,
                         self.masking.cols,
                         :]

        pred_loss = self.loss_fn(z_decoded, y)
        re_loss = self.loss_fn(z_ae_proj, y)
        loss = pred_loss + rnn_loss + re_loss

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
            if metric.name == "rnn":
                metric.update_state(rnn_loss)
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
        # forward in time
        timeseries.reverse()

        # remove current lookback, keep only past samples
        current_lb = ops.squeeze(timeseries.pop())
        ae_projection = self.decoder(self.encoder(current_lb))

        # encode timeseries
        encoded_series = [self.encoder(ops.squeeze(sample))
                          for sample in timeseries]

        # prediction = RNNLayer('simpleRNN')(encoded_series)
        prediction = RNNLayer(self.predictor)(encoded_series)

        prediction_decoded = self.decoder(prediction)
        outputs = {
            'decoded': prediction_decoded,
            'ls_pred': prediction,
            'ae_proj': ae_projection,
        }

        return inputs, outputs


class RNNLayer(layers.Layer):
    def __init__(
            self,
            mode,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode

    def build(self, input_shape):
        dims = input_shape[0][1:]  # ignore batch dim

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

    def call(self, inputs):
        x = self.input_transf(inputs)
        x = self.model(x)
        out = self.output_transf(x)
        return out


class FlattenAndStack(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return ops.stack([layers.Flatten()(sample)
                          for sample in x],
                         axis=1)
