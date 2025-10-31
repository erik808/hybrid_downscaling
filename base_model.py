import keras
from keras import ops
import numpy as np


class BaseModel(keras.Model):
    def __init__(
            self,
            data_gen,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_gen = data_gen

        # use random batch as test input
        idx = np.random.randint(self.data_gen.__len__())
        self.test_x, self.test_y = data_gen.__getitem__(idx)

        self.input_name_LR = 'LR_data'
        self.input_name_HR = 'HR_data'
        self.input_shape_LR = \
            self.test_x[self.input_name_LR].shape[1:]
        self.input_shape_HR = \
            self.test_x[self.input_name_HR].shape[1:]
        self.num_vars = self.input_shape_LR[-1]
        assert self.num_vars == self.input_shape_LR[-1], \
            "unequal variables in LR and HR data"

        # get mask
        self.mask = \
            ops.convert_to_tensor(self.test_x['meta']['mask'][0,])
        self.mask_rows, self.mask_cols = ops.where(self.mask==1)

        self.mask = ops.tile(ops.expand_dims(self.mask, -1),
                             self.num_vars)

        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    def build_model(self, name):
        inputs, outputs = self.builder()
        self.model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=name)
        self.model.build(self.test_x)
        self.build(self.test_x)
        return self.model

    def summary(self):
        return self.model.summary()

    def call(self, inputs, training=True):
        return self.model(inputs, training=training)

    def builder(self):
        pass

    def train_step(self, data, training=True):
        pass

    def test_step(self, data):
        return self.train_step(data, training=False)
