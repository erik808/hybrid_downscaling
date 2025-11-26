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

        self.kernel_regularizer = keras.regularizers.L2(1e-2)

        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.re_loss_tracker = keras.metrics.Mean(name="recons")
        self.KL_loss_tracker = keras.metrics.Mean(name="KLloss")

        # disable layers when bypass enabled
        self.num_layers = 0 if self.bypass_vae else self.num_layers

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
            filters=self.input_filters,
            strides=1,
            kernel_size=3,
            padding='same',
            kernel_initializer="glorot_uniform",
            activation=None,
            name='vae_input_transform',
        )(input_k)
        x = self.create_activation(x)

        # Downsampling layers ---------------------------------
        for i in range(self.num_layers):
            # doubling #filters when used for mean and logvar
            mult = 2 if (i == self.num_layers - 1 and
                         self.sampling_type == 'spatial' and
                         not self.deterministic_mode) else 1

            x = self.conv_downsampling(x, mult)

        if self.bypass_vae:
            x = layers.Identity(name='vae_input_transform')(input_k)

        # Sampling layers ---------------------------------
        if self.sampling_type == 'dense':
            x_shape = x.shape[1:]
            x = layers.Flatten()(x)
            x = layers.Dense(
                units=self.dense_units,
                activation=None,
                kernel_initializer="identity",
                kernel_regularizer=self.kernel_regularizer,
                # trainable=False,
            )(x)
            x = self.create_activation(x)
            mult = 1 if self.deterministic_mode else 2
            x = layers.Dense(
                units=self.latent_space * mult,
                activation=None,
                kernel_initializer="identity",
                kernel_regularizer=self.kernel_regularizer,
                # trainable=False,
            )(x)
            x = self.create_activation(x)

        mean, logvar = Split(
            name="vae_splitter",
            bypass=self.deterministic_mode
        )(x)
        x = Sampling(
            name="vae_sampling",
            bypass=self.deterministic_mode
        )(mean, logvar)

        if self.sampling_type == 'dense':
            x = layers.Dense(
                units=self.dense_units,
                activation=None,
                kernel_initializer="identity",
                kernel_regularizer=self.kernel_regularizer,
                # trainable=False,
            )(x)
            x = self.create_activation(x)
            x = layers.Dense(
                units=ops.prod(x_shape),
                activation=None,
                kernel_initializer="identity",
                kernel_regularizer=self.kernel_regularizer,
                # trainable=False,
            )(x)
            x = self.create_activation(x)
            x = layers.Reshape(x_shape)(x)

        # Upsampling layers ---------------------------------
        y = x
        for i in range(self.num_layers, 0, -1):
            y = self.conv_upsampling(y, i)

        # layer to couple to other models
        y = layers.Conv2D(filters=self.num_filters_hybrid,
                          kernel_size=3,
                          padding='same',
                          name='vae_hybrid_coupling',
                          activation=None,
                          )(y)
        y = self.create_activation(y)

        # output transform
        z = layers.Conv2D(
            filters=self.num_vars,
            kernel_size=3,
            padding='same',
            kernel_initializer="glorot_uniform",
            name='vae_output_conv',
            activation=None)(y)
        z = self.create_activation(z, 'out')

        if self.bypass_vae:
            z = layers.Identity(name='vae_output_conv')(y)

        # activation and masking
        z = self.masking(z)

        outputs = {'decoded': z,
                   'mean': mean,
                   'logvar': logvar,
                   }
        inputs = {self.input_name_HR: inputs}
        return inputs, outputs

    def conv_downsampling(
            self,
            inputs,
            multiple=1,
            version='v2',
            num_layers=1,
            use_residual=True,
    ):
        if version == 'v1':
            out = layers.Conv2D(
                filters=self.filters * multiple,
                strides=2,
                kernel_size=3,
                padding='same',
                activation=None,
            )(inputs)
            return self.create_activation(out)

        elif version == 'v2':
            if (
                    inputs.shape[-1] != self.filters and
                    use_residual and
                    num_layers > 1
            ):
                inputs = layers.Conv2D(
                    filters=self.filters,
                    strides=1,
                    kernel_size=3,
                    padding='same',
                    activation=None,
                )(inputs)

            skip = inputs
            out = inputs
            for i in range(num_layers - 1):
                out = layers.Conv2D(
                    filters=self.filters,
                    strides=1,
                    kernel_size=3,
                    padding='same',
                    activation=None,
                )(out)
                out = self.create_activation(out)

            if use_residual and num_layers > 1:
                out = layers.Add()([out, skip])

            out = layers.Conv2D(
                filters=self.filters,
                strides=2,
                kernel_size=3,
                padding='same',
                activation=None,
            )(out)
            out = self.create_activation(out)
            if multiple > 1:
                out = layers.Conv2D(
                    filters=self.filters * multiple,
                    strides=1,
                    kernel_size=3,
                    padding='same',
                    activation=None,
                )(out)
                out = layers.BatchNormalization()(out)
                out = self.create_activation(out)
            return out

    def conv_upsampling(
            self,
            inputs,
            multiple=1,
    ):
        if self.upsampling_method == 'subpixel':
            return resnet_model.SubPixelConv(
                filters_out=self.filters,
                kernel_size=3,
                scale=2,
                activation=self.activation
            )(inputs)
        elif self.upsampling_method == 'bilinear':
            return resnet_model.UpSampling(
                filters=self.filters,
                kernel_size=3,
                scale=2,
                activation=self.activation,
                method='bilinear',
                num_layers=1,
                use_residual=False,
            )(inputs)
        else:
            raise Exception('invalid upsampling method')

    def create_activation(self, inputs, mode='normal'):
        activation = self.activation_out if mode == 'out' else self.activation
        return base_model.Activation(activation)(inputs)


class Split(layers.Layer):
    """
    Split layer (wrapping ops call as a layer)
    """

    def __init__(
            self,
            bypass=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.bypass = bypass

    def call(self, x):
        if not self.bypass:
            return ops.split(x, 2, axis=-1)
        else:
            return (x, x)


class Sampling(layers.Layer):
    """
    Sampling layer, Gaussian, logvar formulation
    """

    def __init__(
            self,
            bypass=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.bypass = bypass

    def call(self, mean, log_var):
        if not self.bypass:
            eps = keras.random.normal(
                shape=ops.shape(mean)
            )
            out = mean + ops.exp(0.5 * log_var) * eps
        else:
            out = mean
        return out
