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

        resnet_input = resnet_model.get_layer("ResNet").input
        resnet_output = \
            resnet_model.get_layer("ResNet")\
                        .get_layer('resnet_output_conv')\
                        .input

        predictor_input = \
            predictor_model.get_layer('predictor').input
        predictor_output = \
            predictor_model.get_layer('predictor')\
                           .get_layer('decoder')\
                           .get_layer('vae_output_conv').input

        self.resnet_model = keras.Model(
            inputs=resnet_input,
            outputs=resnet_output,
            name="ResNet",
        )

        self.predictor_model = keras.MOdel(
            inputs=predictor_input,
            outputs=predictor_output,
            name="predictor",
        )

        self.resnet_model.trainable = self.trainable_resnet
        self.predictor_model.trainable = self.trainable_predictor

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)
