import keras
from keras import layers
from keras import ops
import tools
import numpy as np
import importlib
import data_manager_cmems
import data_generator_cmems

importlib.reload(data_manager_cmems)
importlib.reload(data_generator_cmems)


class ResNet(keras.Model):

    def __init__(
            self,
            data_gen,
            **kwargs,
    ):
        super().__init__(**kwargs)
        tools.load_config(self, config_name='resnet_model')

        idx = np.random.randint(data_gen.__len__())
        self.test_x, self.test_y = data_gen.__getitem__(idx)
        self.input_name = 'LR_data'
        self.input_shape = self.test_x[self.input_name].shape[1:]
        self.num_vars = self.input_shape[-1]

        # number of necessary upsampling blocks is inferred from LR
        # and HR grids
        grid_HR_shape = self.test_x['meta']['grid_HR']['lat'].shape
        grid_LR_shape = self.test_x['meta']['grid_LR']['lat'].shape
        coarsening = \
            np.asarray(grid_HR_shape) / np.asarray(grid_LR_shape)
        assert coarsening[0] == coarsening[1], "unequal lat/lon coarsening"
        self.coarsening_factor = coarsening[0]
        self.sub_pixel_blocks = int(np.log2(self.coarsening_factor))

    def build_model(self):
        inputs, outputs = self.builder()
        self.model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="ResNet")
        self.model.build(self.test_x)
        self.build(self.test_x)
        return self.model

    def call(self, inputs, training=True):
        return self.model(inputs, training=training)

    def train_step(self, data, training=True):
        pass

    def test_step(self, data):
        return self.train_step(data, training=False)

    def builder(self):
        inputs = layers.Input(
            shape=self.input_shape,
            name=self.input_name)

        # take only the first (newest/current) lookback (axis=1)
        input_k = ops.squeeze(
            ops.split(
                inputs, self.input_shape[0], axis=1)[0])

        x  = layers.Conv2D(filters=64,
                           kernel_size=9,
                           padding='same',
                           activation=None,
                           )(input_k)

        y = layers.PReLU()(x)

        for rs_block in range(self.residual_blocks):
            y = self.residual_block(y)

        # TODO # missing here: Conv - BN - Add

        for sp_block in range(self.sub_pixel_blocks):
            y = self.sub_pixel_convolution(y, scale=2)

        breakpoint()

        return inputs, []

    def residual_block(self, inputs):
        # a residual block (Ledig et al. 2017)
        skip = inputs
        x0 = layers.Conv2D(filters=64,
                           kernel_size=3,
                           padding='same',
                           activation=None,
                           )(inputs)

        x1 = layers.BatchNormalization()(x0)
        x2 = layers.PReLU()(x1)
        x3 = layers.Conv2D(filters=64,
                           kernel_size=3,
                           padding='same',
                           activation=None,
                           )(x2)

        x4 = layers.BatchNormalization()(x3)
        x5 = layers.PReLU()(x4)
        return layers.Add()([x5, skip])

    def sub_pixel_convolution(self, inputs, scale=2):
        # pixel shuffling block (Shi et al. 2016)
        s0 = layers.Conv2D(
            filters=self.num_vars * scale**2,
            kernel_size=3,
            padding='same',
            activation=None,
        )(inputs)

        _, M, N, C = s0.shape
        s1 = layers.Reshape((M, N, scale, scale, self.num_vars))(s0)
        s2 = layers.Permute((1, 3, 2, 4, 5))(s1)
        s3 = layers.Reshape((M * scale, N * scale, self.num_vars))(s2)
        out = layers.PReLU()(s3)
        return out


dmgr_cmems = data_manager_cmems.DataManagerCMEMS()
dmgr_cmems.create_training_data(force_rebuild=False)

dgen_cmems = data_generator_cmems.DataGeneratorCMEMS(
    dm=dmgr_cmems,
    batch_size=4,
    lookback=4,
    mode='train',
    shuffle=True,
    # use_multiprocessing=True,
    # workers=4,
    # max_queue_size=10,
)
resnet = ResNet(dgen_cmems)
resnet.build_model()
