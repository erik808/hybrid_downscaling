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
        self.re_loss_tracker = keras.metrics.Mean(name="recons")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.re_loss_tracker,
        ]

    def train_step(self, data, training=True):
        x, y = data
        if training:
            self.zero_grad()

        z = self({'HR_data': ops.nan_to_num(x['HR_data'])},
                 training=training)

        z = z[:,
              self.masking.rows,
              self.masking.cols,
              :]

        y = y['HR_data'][:,
                         0,  # current lookback index
                         self.masking.rows,
                         self.masking.cols,
                         :]

        re_loss = self.loss_fn(z, y)
        loss = re_loss

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
            if metric.name == "recons":
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

        # lookback ordering is back in time, reversing
        timeseries.reverse()

        # remove current lookback, keep only past samples
        timeseries.pop()

        encoded_series = [self.encoder(ops.squeeze(sample))
                          for sample in timeseries]

        encoded_dims = encoded_series[0].shape

        encoded = layers.Flatten()(ops.stack(encoded_series, axis=-1))
        prediction = layers.Dense(
            units=128,
            activation='leaky_relu')(encoded)
        prediction = layers.Dense(
            units=np.prod(encoded_dims[1:]),
            activation='leaky_relu')(prediction)
        prediction = layers.Reshape(encoded_dims[1:])(prediction)
        prediction_decoded = self.decoder(prediction)
        outputs = self.masking(prediction_decoded)
        return inputs, outputs
