import keras
from keras import ops
from keras import layers
from keras import backend as K
import numpy as np

K.clear_session()


class BaseModel(keras.Model):
    def __init__(
            self,
            data_gen,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_gen = data_gen

        # use random batch as test input
        test_x, test_y = self.get_random_item()

        self.coarsening_factor = self.get_coarsening_factor(test_x)

        self.input_name_LR = 'LR_data'
        self.input_name_HR = 'HR_data'
        self.input_shape_LR = \
            test_x[self.input_name_LR].shape[1:]
        self.input_shape_HR = \
            test_x[self.input_name_HR].shape[1:]
        self.num_vars = self.input_shape_LR[-1]
        assert self.num_vars == self.input_shape_LR[-1], \
            "unequal variables in LR and HR data"
        # create mask
        self.masking = \
            Masking(test_x['meta']['mask'][0,],
                    self.num_vars,
                    name=self.name + '_masking')

        # setup loss and loss tracker
        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    def get_random_item(self):
        idx = np.random.randint(self.data_gen.__len__())
        return self.data_gen.__getitem__(idx)

    def build_model(self, name):
        inputs, outputs = self.builder()
        self.model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=name)

        # use random batch as test input
        test_x, test_y = self.get_random_item()
        self.build(test_x)
        return self.model

    def build(self, *args, **kwargs):
        self.model.build(*args, **kwargs)

    def get_coarsening_factor(self, test_x):
        # number of necessary upsampling blocks is inferred from LR
        # and HR grids
        grid_HR_shape = test_x['meta']['grid_HR']['lat'].shape
        grid_LR_shape = test_x['meta']['grid_LR']['lat'].shape
        coarsening = \
            np.asarray(grid_HR_shape) / np.asarray(grid_LR_shape)
        assert coarsening[0] == coarsening[1], "unequal lat/lon coarsening"
        coarsening_factor = coarsening[0]
        return coarsening_factor

    def summary(self, **kwargs):
        return self.model.summary(**kwargs)

    def call(self, inputs, training=True):
        return self.model(inputs, training=training)

    def builder(self):
        pass

    def train_step(self, data, training=True):
        pass

    def test_step(self, data):
        return self.train_step(data, training=False)


class Masking(layers.Layer):
    def __init__(
            self,
            mask,
            num_vars,
            **kwargs,
    ):
        super().__init__(**kwargs)
        mask = ops.convert_to_tensor(mask)
        self.rows, self.cols = ops.where(mask==1)
        self.num_vars = num_vars

        # usable mask for multiply
        self.mask = ops.tile(
            ops.expand_dims(mask, -1),
            self.num_vars)

    def call(self, inputs):
        return ops.multiply(inputs, self.mask)
