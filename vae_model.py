import keras
from keras import layers
from keras import ops
# from keras import regularizers
import torch
import tools
import base_model
import resnet_model
import importlib

importlib.reload(base_model)
importlib.reload(resnet_model)


class VAE(base_model.BaseModel):

    def __init__(
            self,
            **kwargs,
    ):

        super().__init__(**kwargs)
        tools.load_config(self, config_name='vae_model')

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

    def create_input(self, x):
        return {'HR_data': ops.nan_to_num(x['HR_data'])}

    def train_step(self, data, training=True):
        x, y = data
        if training:
            self.zero_grad()

        z = self(self.create_input(x), training=training)

        z_decoded = z['decoded']
        z_mean = z['mean']
        z_logvar = z['logvar']

        # compute reconstruction loss
        z_decoded = z_decoded[:,
                              self.masking.rows,
                              self.masking.cols,
                              :]
        y = y['HR_data'][:,
                         0,
                         self.masking.rows,
                         self.masking.cols,
                         :]

        re_loss = self.loss_fn(z_decoded, y) * self.gamma

        # kl loss variance formulation
        kl_loss = self.loss_KL(z_mean, z_logvar, beta=self.beta)

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

    def loss_KL(self, mean, logvar, beta=1):
        """Kullback Leibler loss, Gaussian prior, logvar formulation"""
        return \
            -beta / 2 * \
            ops.mean(1 + logvar -
                     ops.exp(logvar) -
                     ops.square(mean))

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

        # Input transform
        x = layers.Conv2D(
            filters=32,
            strides=1,
            kernel_size=9,
            padding='same',
            activation=None,
            name='vae_input_transform',
        )(input_k)
        x = self.create_activation(x)

        # Downsampling layers
        for i in range(self.num_layers):
            # doubling #filters when used for mean and logvar
            mult = 2 if (i == self.num_layers - 1 and
                         not self.deterministic_mode) else 1

            x = self.conv_downsampling(x, mult)

        # Sampling layer
        if not self.deterministic_mode:
            mean, logvar = Split(name="vae_splitter")(x)
            x = Sampling(name="vae_sampling")(mean, logvar)
        else:
            mean = x
            logvar = x

        # Upsampling layers
        y = x
        for i in range(self.num_layers):
            y = self.conv_upsampling(y)

        # connection to skip the output convolution
        skip_output = layers.Identity(name='skip_output')(y)

        # output transform
        y = layers.Conv2D(
            filters=self.num_vars,
            kernel_size=9,
            padding='same',
            name='vae_output_conv',
            activation='sigmoid')(y)

        # activation and masking
        z = self.masking(y)

        outputs = {'decoded': z,
                   'mean': mean,
                   'logvar': logvar,
                   'skip_output': skip_output,
                   }
        inputs = {self.input_name_HR: inputs}
        return inputs, outputs

    def conv_downsampling(
            self,
            inputs,
            multiple=1
    ):
        out = layers.Conv2D(
            filters=self.filters * multiple,
            strides=2,
            kernel_size=3,
            padding='same',
            activation=None,
        )(inputs)
        return self.create_activation(out)

    def conv_upsampling(
            self,
            inputs,
    ):
        return resnet_model.SubPixelConv(
            filters_out=self.filters,
            kernel_size=3,
            scale=2,
            activation=self.activation
        )(inputs)

    def create_activation(self, inputs):
        if self.activation == 'prelu':
            return layers.PReLU()(inputs)
        else:
            return layers.Activation(self.activation)(inputs)


class Split(layers.Layer):
    """
    Split layer (wrapping ops call as a layer)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return ops.split(x, 2, axis=-1)


class Sampling(layers.Layer):
    """
    Sampling layer, Gaussian, logvar formulation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, mean, log_var):
        eps = keras.random.normal(
            shape=ops.shape(mean)
        )
        out = mean + ops.exp(0.5 * log_var) * eps
        return out
