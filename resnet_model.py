import torch
import keras
from keras import layers
from keras import ops
import tools
import importlib
import numpy as np
import base_model

importlib.reload(base_model)


class ResNet(base_model.BaseModel):

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

        tools.load_config(self, config_name='resnet_model')

        self.upsampling_blocks = int(np.log2(self.coarsening_factor))

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)

    def create_input(self, inputs):
        return {'LR_data': inputs['LR_data']}

    def train_step(self, data, training=True):
        x, y = data
        if training:
            self.zero_grad()

        z = self(self.create_input(x), training=training)
        y = y['HR_data'][:,
                         0,
                         self.masking.rows,
                         self.masking.cols,
                         :]
        z = z[:,
              self.masking.rows,
              self.masking.cols,
              :]

        loss = self.loss_MSLE(z, y)

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
        """builder uses the functional api, some separate blocks follow
        subclassing style

        """
        # ResNet will take LR input
        inputs = layers.Input(
            shape=self.input_shape_LR,
            name=self.input_name_LR)

        # take only the first (newest/current) lookback (axis=1)
        input_k = ops.squeeze(
            ops.split(
                inputs,
                self.input_shape_LR[0],
                axis=1)[0],
            axis=1)  # squeeze only axis=1 to support batch_size=1

        y = InputTransform(
            filters=self.num_filters,
            kernel_size=9,
            activation=self.activation,
        )(input_k)

        skip = y

        # residual blocks
        for rs_block in range(self.residual_blocks):
            y = ResidualBlock(
                filters=self.num_filters,
                kernel_size=3,
                activation=self.activation,
            )(y)

        # Conv - BN - Add
        y = layers.Conv2D(filters=self.num_filters,
                          kernel_size=3,
                          padding='same',
                          activation=None,
                          )(y)
        y = layers.BatchNormalization()(y)
        y = layers.Add()([y, skip])

        # subpixel convolutions
        for ups_block in range(self.upsampling_blocks):
            if self.upsampling_method == 'subpixel':
                y = SubPixelConv(
                    filters_out=self.num_filters,
                    kernel_size=3,
                    activation=self.activation,
                    scale=2,
                )(y)
            elif self.upsampling_method == 'bilinear':
                y = UpSampling(
                    filters=self.num_filters,
                    kernel_size=3,
                    activation=self.activation,
                    scale=2,
                    method='bilinear'
                )(y)
            else:
                raise Exception('invalid upsampling method')

        # layers to couple to other models
        y = layers.Conv2D(filters=self.num_filters_hybrid,
                          kernel_size=3,
                          padding='same',
                          name='hybrid_coupling',
                          activation=None,
                          )(y)
        y = base_model.Activation(self.activation,
                                  name='hybrid_coupling_actv')(y)

        outputs = OutputBlock(
            num_filters=self.num_filters_hybrid,
            num_filters_out=self.num_vars,
            kernel_size=3,
            kernel_size_out=9,
            activation=self.activation,
            activation_out=self.activation_out,
            padding='same',
            num_output_layers=self.num_output_layers,
            name="resnet_output_block",
            bypass=False,
        )(y)

        # masking
        outputs = self.masking(outputs)
        return {'LR_data': inputs}, outputs


class OutputBlock(layers.Layer):
    def __init__(
            self,
            num_filters=64,
            num_filters_out=9,
            kernel_size=3,
            kernel_size_out=9,
            activation='prelu',
            activation_out='sigmoid',
            padding='same',
            num_output_layers=2,
            bypass=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_output_layers = num_output_layers
        self.num_filters = num_filters
        self.num_filters_out = num_filters_out
        self.kernel_size = kernel_size
        self.kernel_size_out = kernel_size_out
        self.activation = activation
        self.activation_out = activation_out
        self.padding = padding
        self.bypass = bypass
        self.num_output_layers = 0 if self.bypass else self.num_output_layers

    def build(self, input_shape):
        self.lrs = []
        for i in range(self.num_output_layers):
            self.lrs.append(
                layers.Conv2D(filters=self.num_filters,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              name='l' + str(i),
                              activation=None,
                              )
            )
            self.lrs.append(base_model.Activation(self.activation))

        self.output_conv = layers.Conv2D(
            filters=self.num_filters_out,
            kernel_size=self.kernel_size_out,
            padding=self.padding,
            name='output_block_last',
        )

        # Todo # Different output activations
        # should be tested. Output values need
        # to be mapped to [0,1].
        self.output_actv = base_model.Activation(self.activation_out)

        if self.bypass:
            self.output_conv = layers.Identity()

    def call(self, inputs):
        for layer in self.lrs:
            inputs = layer(inputs)
        inputs = self.output_conv(inputs)
        return self.output_actv(inputs)


class InputTransform(layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            activation='prelu',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def build(self, input_shape):
        self.conv = layers.Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  padding='same',
                                  activation=None,
                                  )
        self.actv = base_model.Activation(self.activation)

    def call(self, inputs):
        x = self.conv(inputs)
        return self.actv(x)


class ResidualBlock(layers.Layer):
    """ a residual block (Ledig et al. 2017) """
    def __init__(
            self,
            filters,
            kernel_size,
            activation='prelu',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def build(self, input_shape):
        self.conv2d_1 = layers.Conv2D(filters=self.filters,
                                      kernel_size=self.kernel_size,
                                      padding='same',
                                      activation=None,
                                      )
        self.BN_1 = layers.BatchNormalization()
        self.actv_1 = base_model.Activation(self.activation)
        self.conv2d_2 = layers.Conv2D(filters=self.filters,
                                      kernel_size=self.kernel_size,
                                      padding='same',
                                      activation=None,
                                      )
        self.BN_2 = layers.BatchNormalization()
        self.actv_2 = base_model.Activation(self.activation)
        self.add = layers.Add()

    def call(self, inputs):
        skip = inputs
        x = self.conv2d_1(inputs)
        x = self.BN_1(x)
        x = self.actv_1(x)
        x = self.conv2d_2(x)
        x = self.BN_2(x)
        x = self.actv_2(x)
        return self.add([x, skip])


class SubPixelConv(layers.Layer):
    """ pixel shuffling block (Shi et al. 2016) """
    def __init__(
            self,
            filters_out,
            kernel_size,
            activation='prelu',
            scale=2,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters_out = filters_out
        self.kernel_size = kernel_size
        self.scale = scale
        self.activation = activation

    def build(self, input_shape):
        self.conv2d = layers.Conv2D(
            filters=self.filters_out * self.scale**2,
            kernel_size=self.kernel_size,
            padding='same',
            activation=None,
        )

        _, M, N, C = input_shape
        self.reshape1 = layers.Reshape(
            (M, N, self.scale, self.scale, self.filters_out))
        self.permute = layers.Permute((1, 3, 2, 4, 5))
        self.reshape2 = layers.Reshape(
            (M * self.scale, N * self.scale, self.filters_out))

        self.actv = base_model.Activation(self.activation)

    def call(self, inputs):
        s = self.conv2d(inputs)
        s = self.reshape1(s)
        s = self.permute(s)
        s = self.reshape2(s)
        return self.actv(s)


class UpSampling(layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            activation='prelu',
            method='bilinear',
            scale=2,
            num_layers=1,
            use_residual=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.scale = scale
        self.method = method
        self.activation = activation
        self.num_layers = num_layers
        self.use_residual = use_residual

    def build(self, input_shape):
        _, M, N, C = input_shape

        self.upsample = layers.UpSampling2D(
            size=(self.scale, self.scale),
            interpolation=self.method)

        self.conv_layers = []
        self.BN_layers = []
        self.actv_layers = []
        for cl in range(self.num_layers):
            self.conv_layers.append(
                layers.Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    padding='same',
                    activation=None,
                )
            )
            # self.BN_layers.append(layers.BatchNormalization())
            self.actv_layers.append(base_model.Activation(self.activation))

        self.add = layers.Add()

    def call(self, inputs):
        s = self.upsample(inputs)
        skip = s
        for i in range(self.num_layers):
            s = self.conv_layers[i](s)
            # s = self.BN_layers[i](s)
            s = self.actv_layers[i](s)

        out = s
        if self.use_residual:
            out = self.add([out, skip])

        return out
