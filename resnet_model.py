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

        self.get_coarsening_factor()

        self.sub_pixel_blocks = int(np.log2(self.coarsening_factor))

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)

    def get_coarsening_factor(self):
        # number of necessary upsampling blocks is inferred from LR
        # and HR grids
        grid_HR_shape = self.test_x['meta']['grid_HR']['lat'].shape
        grid_LR_shape = self.test_x['meta']['grid_LR']['lat'].shape
        coarsening = \
            np.asarray(grid_HR_shape) / np.asarray(grid_LR_shape)
        assert coarsening[0] == coarsening[1], "unequal lat/lon coarsening"
        self.coarsening_factor = coarsening[0]

    def train_step(self, data, training=True):
        x, y = data
        if training:
            self.zero_grad()

        z = self({'LR_data': x['LR_data']}, training=training)
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

        y = InputTransform(filters=self.num_filters,
                           kernel_size=9)(input_k)

        skip = y

        # residual blocks
        for rs_block in range(self.residual_blocks):
            y = ResidualBlock(
                filters=self.num_filters,
                kernel_size=3,
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
        for sp_block in range(self.sub_pixel_blocks):
            y = SubPixelConv(
                filters_out=self.num_filters,
                kernel_size=3,
                scale=2,
            )(y)

        outputs = layers.Conv2D(filters=self.num_vars,
                                kernel_size=9,
                                padding='same',
                                # Todo # Different output activations
                                # should be tested. Output values need
                                # to be mapped to [0,1].
                                activation='sigmoid',
                                )(y)

        # masking
        outputs = ops.multiply(outputs, self.mask)

        return inputs, outputs


class InputTransform(layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  padding='same',
                                  activation=None,
                                  )
        self.actv = layers.PReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        return self.actv(x)


class ResidualBlock(layers.Layer):
    """ a residual block (Ledig et al. 2017) """
    def __init__(
            self,
            filters,
            kernel_size,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv2d_1 = layers.Conv2D(filters=self.filters,
                                      kernel_size=self.kernel_size,
                                      padding='same',
                                      activation=None,
                                      )
        self.BN_1 = layers.BatchNormalization()
        self.actv_1 = layers.PReLU()
        self.conv2d_2 = layers.Conv2D(filters=self.filters,
                                      kernel_size=self.kernel_size,
                                      padding='same',
                                      activation=None,
                                      )
        self.BN_2 = layers.BatchNormalization()
        self.actv_2 = layers.PReLU()
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
            scale,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters_out = filters_out
        self.kernel_size = kernel_size
        self.scale = scale

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
        self.actv = layers.PReLU()

    def call(self, inputs):
        s = self.conv2d(inputs)
        s = self.reshape1(s)
        s = self.permute(s)
        s = self.reshape2(s)
        return self.actv(s)
