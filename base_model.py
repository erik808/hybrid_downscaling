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

        # get mask
        self.mask = \
            ops.convert_to_tensor(self.test_x['meta']['mask'][0,])
        self.mask_rows, self.mask_cols = ops.where(self.mask==1)

        self.mask = ops.tile(ops.expand_dims(self.mask, -1),
                             self.num_vars)
