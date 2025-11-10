import torch
import keras
from keras import ops
from keras import layers

import numpy as np
import importlib
import tools
import base_model
import resnet_model as rm
import predictor_model as pm

importlib.reload(base_model)
importlib.reload(rm)
importlib.reload(pm)


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

        predictor_model.get_layer('predictor').layers

        self.resnet_model = keras.Model(
            inputs=resnet_input,
            outputs=resnet_output,
            name="ResNet",
        )

        self.predictor_model = keras.Model(
            inputs=predictor_input,
            outputs=predictor_output,
            name="predictor",
        )

        self.hybrid_HR_input = self.input_name_HR + '_hybrid'
        self.hybrid_LR_input = self.input_name_LR + '_hybrid'

        self.resnet_model.trainable = self.trainable_resnet
        self.predictor_model.trainable = self.trainable_predictor

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    def train_step(self, data, training=True):
        x, y = data
        if training:
            self.zero_grad()

        breakpoint()
        z = self.model({self.input_name_HR:
                        ops.nan_to_num(x[self.input_name_HR]),
                        self.input_name_LR:
                        ops.nan_to_num(x[self.input_name_LR])},
                       training=training)

    def builder(self):
        input_HR = layers.Input(
            shape=self.input_shape_HR,
            name=self.input_name_HR)

        input_LR = layers.Input(
            shape=self.input_shape_LR,
            name=self.input_name_LR)

        resnet_result = self.resnet_model(input_LR)
        predictor_result = self.predictor_model(input_HR)

        # if self.hybridization == 'product':
        assert resnet_result.shape == predictor_result.shape, \
            "implement some reshaping/upsampling/downsampling here"
        x = layers.Multiply()([resnet_result, predictor_result])

        x = rm.ResidualBlock(
            filters=64,
            kernel_size=3,
        )(x)

        out = layers.Conv2D(
            filters=self.num_vars,
            kernel_size=9,
            padding='same',
            name='hybrid_output_conv',
            activation='sigmoid')(x)

        out = self.masking(out)
        outputs = out

        inputs = {self.input_name_HR: input_HR,
                  self.input_name_LR: input_LR}

        return inputs, outputs
