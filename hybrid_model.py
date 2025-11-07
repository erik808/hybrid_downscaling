import torch
import keras
from keras import ops
from keras import layers

import numpy as np
import importlib
import tools
import base_model
import resnet_model
import predictor_model

importlib.reload(base_model)
importlib.reload(resnet_model)
importlib.reload(predictor_model)


class Hybrid(base_model.BaseModel):
    def __init__(
            self,
            resnet_model,
            predictor_model,
            **kwargs,
    ):
        super().__init__(**kwargs)
        tools.load_config(self, config_name='hybrid_model')
        self.resnet_model = resnet_model
        self.predictor_model = predictor_model


        self.resnet_model.get_layer("ResNet")\
                         .get_layer('resnet_output_conv').input.shape

        breakpoint()
        self.predictor_model.get_layer("predictor")\
                        .get_layer('predictor_output_conv').input.shape

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)
