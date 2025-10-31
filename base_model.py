import keras
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
