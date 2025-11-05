import keras
from keras import layers
from keras import ops
# from keras import regularizers
import torch
import tools
import base_model


class VAE(base_model.BaseModel):

    def __init__(
            self,
            **kwargs,
    ):

        super().__init__(**kwargs)
        tools.load_config(self, config_name='vae_model')

        self.loss_fn = keras.losses.MeanSquaredError()

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.re_loss_tracker = keras.metrics.Mean(name="recons")
        self.KL_loss_tracker = keras.metrics.Mean(name="KLloss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.re_loss_tracker,
            self.KL_loss_tracker,
        ]

    def train_step(self, data, training=True):
        x, y = data
        if training:
            self.zero_grad()

        z = self({'HR_data': ops.nan_to_num(x['HR_data'])},
                 training=training)

        z_decoded = z['decoded']
        z_mean = z['mean']
        z_logvar = z['logvar']

        # compute reconstruction loss
        z_decoded = z_decoded[:, self.mask_rows, self.mask_cols, :]
        y = y['HR_data'][:, 0, self.mask_rows, self.mask_cols, :]
        re_loss = self.loss_fn(z_decoded, y) * self.gamma

        # compute KL loss (sigma formulation)
        # kl_loss = \
        #     -self.beta / 2 * \
        #     ops.mean(1 + 2 * z_logsigma -
        #              ops.exp(2 * z_logsigma) -
        #              ops.square(z_mean))

        # kl loss variance formulation
        kl_loss = \
            -self.beta / 2 * \
            ops.mean(1 + z_logvar -
                     ops.exp(z_logvar) -
                     ops.square(z_mean))

        # # sum over latent dimension, mean over batch size
        # kl_loss = ops.mean(kl_loss)

        # combine losses
        loss = re_loss + kl_loss

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
            if metric.name == "KLloss":
                metric.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def builder(self):
        pad_before = ((0, 0), (1, 0), (1, 0), (0, 0))
        pad_after = ((0, 0), (0, 1), (0, 1), (0, 0))
        crop_before = ((0, 0), (-1, 0), (-1, 0), (0, 0))
        crop_after = ((0, 0), (0, -1), (0, -1), (0, 0))

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
        x = layers.Conv2D(
            filters=self.filter_mult_start * self.input_shape_HR[-1],
            strides=2,
            kernel_size=3,
            padding='valid',
            activation=self.activation,
        )(input_k)
        x = ops.pad(x, pad_before)

        x = layers.Conv2D(
            filters=self.filter_mult_rest * x.shape[-1],
            strides=2,
            kernel_size=3,
            padding='valid',
            activation=self.activation,
        )(x)
        x = ops.pad(x, pad_after)

        x = layers.Conv2D(
            self.filter_mult_rest * x.shape[-1],
            strides=2,
            kernel_size=3,
            padding='valid',
            activation=self.activation,
        )(x)
        x = ops.pad(x, pad_before)

        x = layers.Conv2D(
            self.filter_mult_rest * x.shape[-1],
            strides=2,
            kernel_size=3,
            padding='valid',
            activation=self.activation,
        )(x)
        x = ops.pad(x, pad_after)
        skip = x

        # return to this shape for decoder input
        # return_shape = x.shape
        # skip = x  # in deterministic mode

        # # dense transform
        # # x = layers.Flatten()(x)

        # y = layers.Dense(units=self.dense_dim,
        #                  activation=self.activation,
        #                  # kernel_initializer="identity",
        #                  )(x)

        # y = layers.Dense(units=self.latent_space_dim * 2,
        #                  activation=None,
        #                  # kernel_initializer="identity",
        #                  name="mean_logvar",
        #                  )(y)

        mean, logvar = ops.split(x, 2, axis=-1)
        # -------------------------------------------------------
        # Sampling
        y = Sampling()(mean, logvar)

        y = layers.Conv2DTranspose(
            filters=int(y.shape[-1] * self.filter_mult_rest),
            strides=1,
            kernel_size=3,
            padding='valid',
            activation=self.activation,
        )(y)
        y = ops.pad(y, crop_before)
        y = ops.pad(y, crop_after)
        # -------------------------------------------------------
        # Decoder
        # z = layers.Dense(
        #     units=self.dense_dim,
        #     activation=self.activation,
        # )(y)

        # z = layers.Dense(
        #     units=ops.prod(return_shape[1:]),
        #     # kernel_initializer="identity",
        #     # trainable=False,
        #     # kernel_regularizer=regularizers.L2(1e-1),
        #     activation=self.activation,
        # )(z)

        # z = layers.Reshape(return_shape[1:])(z)

        # if self.deterministic_mode:
        #     z = skip
        #     mean = skip
        #     logvar = skip

        if self.deterministic_mode:
            y = skip

        z = layers.Conv2DTranspose(
            filters=int(y.shape[-1] / self.filter_mult_rest),
            strides=2,
            kernel_size=3,
            padding='valid',
            activation=self.activation,
        )(y)
        z = ops.pad(z, crop_before)

        z = layers.Conv2DTranspose(
            filters=int(z.shape[-1] / self.filter_mult_rest),
            strides=2,
            kernel_size=3,
            padding='valid',
            activation=self.activation,
        )(z)
        z = ops.pad(z, crop_after)
        z = layers.Conv2DTranspose(
            filters=int(z.shape[-1] / self.filter_mult_rest),
            strides=2,
            kernel_size=3,
            padding='valid',
            activation=self.activation,
        )(z)
        z = ops.pad(z, crop_before)
        z = layers.Conv2DTranspose(
            filters=int(z.shape[-1] / self.filter_mult_start),
            strides=2,
            kernel_size=3,
            padding='valid',
            activation=None,
        )(z)
        z = ops.pad(z, crop_after)
        breakpoint()

        # activation and masking
        z = ops.multiply(z, self.mask)
        outputs = {'decoded': z,
                   'mean': mean,
                   'logvar': logvar,
                   }
        return inputs, outputs


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
