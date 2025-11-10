import torch
import keras
from keras import ops
from keras import layers

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

        self.resnet_model = resnet_model
        self.predictor_model = predictor_model

        self.resnet_input = resnet_model.get_layer("ResNet").input
        self.resnet_output = \
            resnet_model.get_layer("ResNet")\
                        .get_layer('resnet_output_conv')\
                        .input

        self.predictor_input = \
            predictor_model.get_layer('predictor').input

        self.predictor_output = \
            self.predictor_model.get_layer('predictor')\
                                .get_layer('decoder_pred')\
                                .get_layer('vae_output_conv').input

        self.ae_recons = \
            self.predictor_model.get_layer('predictor').output['ae_recons']
        self.ls_pred = \
            self.predictor_model.get_layer('predictor').output['ls_pred']

        self.resnet_layers = keras.Model(
            inputs=self.resnet_input,
            outputs=self.resnet_output,
            name="ResNet_submodel",
        )

        self.predictor_layers = keras.Model(
            inputs=self.predictor_input,
            outputs=self.predictor_output,
            name="predictor_submodel",
        )

        self.resnet_layers.trainable = self.trainable_resnet
        self.predictor_layers.trainable = self.trainable_predictor
        self.trainable_VAE = self.predictor_model.trainable_VAE

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.re_loss_tracker = keras.metrics.Mean(name="reconstruction")
        self.pred_loss_tracker = keras.metrics.Mean(name="prediction")
        self.lspred_loss_tracker = keras.metrics.Mean(name="ls_pred")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.re_loss_tracker,
            self.pred_loss_tracker,
            self.lspred_loss_tracker,
        ]

    def train_step(self, data, training=True):
        x, y = data
        if training:
            self.zero_grad()

        y_ls = \
            self.predictor_model.get_layer('encoder_pred')(
                ops.nan_to_num(
                    ops.squeeze(
                        y[self.input_name_HR][:,
                                              0,  # target lookback index
                                              ...],
                        axis=1)))

        z = self.model({self.input_name_HR:
                        ops.nan_to_num(x[self.input_name_HR]),
                        self.input_name_LR:
                        ops.nan_to_num(x[self.input_name_LR])},
                       training=training)

        z_hybrid = z['hybrid'][:,
                               self.masking.rows,
                               self.masking.cols,
                               :]

        z_ae_recons = z['ae_recons'][:,
                                     self.masking.rows,
                                     self.masking.cols,
                                     :]

        z_ls_pred = z['ls_pred']
        lspred_loss = self.loss_fn(z_ls_pred, y_ls)

        def y_k(k):
            return \
                y[self.input_name_HR][:,
                                      k,  # kth lookback index
                                      self.masking.rows,
                                      self.masking.cols,
                                      :]

        if self.trainable_VAE:
            re_loss = self.loss_fn(z_ae_recons, y_k(1))
        else:
            re_loss = 0.0

        # total loss
        pred_loss = self.loss_fn(z_hybrid, y_k(0))
        loss = pred_loss + re_loss + lspred_loss

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
            if metric.name == "prediction":
                metric.update_state(pred_loss)
            if metric.name == "ls_pred":
                metric.update_state(lspred_loss)
            if metric.name == "reconstruction":
                metric.update_state(re_loss)

        return {m.name: m.result() for m in self.metrics}

    def builder(self):

        # reusing predictor input
        input_HR = self.predictor_input
        assert input_HR.shape[1] > 1, \
            "need at least lookback=2 to make predictions"

        # reusing resnet input
        input_LR = self.resnet_input

        resnet_result = self.resnet_layers(input_LR)
        predictor_result = self.predictor_layers(input_HR)

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

        outputs = {'hybrid': out,
                   'ae_recons': self.ae_recons,
                   'ls_pred': self.ls_pred,
                   }

        inputs = {self.input_name_HR: input_HR,
                  self.input_name_LR: input_LR}

        return inputs, outputs
