import keras
from keras import layers
from keras import ops
import torch
import tools
import base_model
import resnet_model


class VAE(base_model.BaseModel):

    def __init__(
            self,
            **kwargs,
    ):

        super().__init__(**kwargs)
        tools.load_config(self, config_name='vae_model')

        # weight on KL loss
        self.beta = 1e-7

        self.loss_fn = keras.losses.MeanSquaredError()

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def train_step(self, data, training=True):
        x, y = data
        if training:
            self.zero_grad()

        z = self({'HR_data': x['HR_data']}, training=training)
        y = y['HR_data'][:, 0, self.mask_rows, self.mask_cols, :]
        z = z[:, self.mask_rows, self.mask_cols, :]
        loss = self.loss_fn(z, y)

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

        return {m.name: m.result() for m in self.metrics}

    def builder(self):
        inputs = layers.Input(
            shape=self.input_shape_HR,
            name=self.input_name_HR)

        # take only the first (newest/current) lookback (axis=1)
        input_k = ops.squeeze(
            ops.split(
                inputs,
                self.input_shape_LR[0],
                axis=1)[0],
            axis=1)  # squeeze only axis=1 to support batch_size=1

        # -------------------------------------------------------
        # Encoder
        x = layers.Conv2D(filters=4 * self.input_shape_HR[-1],
                          strides=2,
                          kernel_size=3,
                          padding='same',
                          activation=None,
                          )(input_k)
        x = layers.ELU()(x)
        x = layers.Conv2D(filters=4 * x.shape[-1],
                          strides=2,
                          kernel_size=3,
                          padding='same',
                          activation=None,
                          )(x)
        x = layers.ELU()(x)
        x = layers.Conv2D(filters=4 * x.shape[-1],
                          strides=2,
                          kernel_size=3,
                          padding='same',
                          activation=None,
                          )(x)
        x = layers.ELU()(x)
        x = layers.Conv2D(filters=4 * x.shape[-1],
                          strides=2,
                          kernel_size=3,
                          padding='same',
                          activation=None,
                          )(x)
        x = layers.ELU()(x)
        x = layers.Dense(units=256,
                         activation=None,
                         )(x)
        x = layers.ELU()(x)

        mean = layers.Dense(units=self.latent_space_dim,
                            activation=None,
                            name="mean",
                            )(x)

        logsigma = layers.Dense(units=self.latent_space_dim,
                                activation=None,
                                name="logsigma",
                                )(x)

        breakpoint()

        outputs = []
        return inputs, outputs


class Sampling(layers.Layer):
    """
    Sampling layer
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # optional seed here

    def call(self, mean, log_sigma):
        eps = keras.random.normal(
            shape=ops.shape(mean)
        )
        out = mean + ops.exp(log_sigma) * eps
        return out
