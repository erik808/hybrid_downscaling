import keras
from keras import layers
from keras import ops
# from keras import regularizers
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
            filters=32,
            strides=1,
            kernel_size=9,
            padding='same',
            activation=None,
        )(input_k)
        x = layers.PReLU()(x)

        x = layers.Conv2D(
            filters=64,
            strides=2,
            kernel_size=3,
            padding='same',
            activation=None,
        )(input_k)
        x = layers.PReLU()(x)

        x = layers.Conv2D(
            filters=64,
            strides=2,
            kernel_size=3,
            padding='same',
            activation=None,
        )(x)
        x = layers.PReLU()(x)

        x = layers.Conv2D(
            filters=64,
            strides=2,
            kernel_size=3,
            padding='same',
            activation=None,
        )(x)
        x = layers.PReLU()(x)

        x = layers.Conv2D(
            filters=64,
            strides=2,
            kernel_size=3,
            padding='same',
            activation=None,
        )(x)
        x = layers.PReLU()(x)

        if not self.deterministic_mode:
            mean, logvar = ops.split(x, 2, axis=-1)
            # Sampling
            x = Sampling()(mean, logvar)
        else:
            mean = x
            logvar = x

        y = resnet_model.SubPixelConv(
            filters_out=64,
            kernel_size=3,
            scale=2)(x)

        y = resnet_model.SubPixelConv(
            filters_out=64,
            kernel_size=3,
            scale=2)(y)

        y = resnet_model.SubPixelConv(
            filters_out=64,
            kernel_size=3,
            scale=2)(y)

        y = resnet_model.SubPixelConv(
            filters_out=64,
            kernel_size=3,
            scale=2)(y)

        y = layers.Conv2D(
            filters=self.num_vars,
            kernel_size=9,
            padding='same',
            activation='linear')(y)

        # activation and masking
        z = ops.multiply(y, self.mask)

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
