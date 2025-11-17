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

        self.resnet_input = resnet_model.model.input['LR_data']
        self.resnet_output = resnet_model.model.output

        # coupling point where we choose to do the hybridization in
        # resnet
        self.resnet_coupling = \
            resnet_model.model\
                        .get_layer('resnet_output_block')\
                        .input

        self.predictor_input = \
            predictor_model.model.input['HR_data']

        self.predictor_output = \
            predictor_model.model.output['skip_vae_output']

        self.ae_mean = \
            predictor_model.model.output['mean']
        self.ae_logvar = \
            predictor_model.model.output['logvar']
        self.ae_recons = \
            predictor_model.model.output['ae_recons']
        self.ls_pred = \
            predictor_model.model.output['ls_pred']

        self.resnet_layers = keras.Model(
            inputs=self.resnet_input,
            outputs=self.resnet_coupling,
            name="ResNet_submodel",
        )

        self.encoder = predictor_model.model.get_layer('encoder')

        self.predictor_layers = keras.Model(
            inputs=self.predictor_input,
            outputs=self.predictor_output,
            name="predictor_submodel",
        )

        self.output_block = keras.Model(
            inputs=self.resnet_coupling,
            outputs=self.resnet_output,
            name="output_block",
        )

        self.resnet_layers.trainable = self.trainable_resnet
        # Only allow disabling of predictor. Enabling would also
        # enable possibly disabled encoder and decoder.
        if not self.trainable_predictor:
            self.predictor_layers.trainable = self.trainable_predictor

        self.trainable_VAE = predictor_model.trainable_VAE
        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_fn_KL = predictor_model.loss_KL

        self.trackers = []
        self.trackers.append(keras.metrics.Mean(name="loss"))
        if 'reconstruction' in self.loss_list:
            self.trackers.append(keras.metrics.Mean(name="recon"))
        if 'KL' in self.loss_list:
            self.trackers.append(keras.metrics.Mean(name="KL"))
        if 'outer_pred' in self.loss_list:
            self.trackers.append(keras.metrics.Mean(name="outer_pred"))
        if 'inner_pred' in self.loss_list:
            self.trackers.append(keras.metrics.Mean(name="inner_pred"))

    @property
    def metrics(self):
        return self.trackers

    def create_input(self, inputs):
        return {self.input_name_HR:
                ops.nan_to_num(inputs[self.input_name_HR]),
                self.input_name_LR:
                ops.nan_to_num(inputs[self.input_name_LR])}

    def train_step(self, data, training=True):
        x, y = data
        if training:
            self.zero_grad()

        z = self.model(self.create_input(x), training=training)

        # latent space prediction
        if 'inner_pred' in self.loss_list:
            z_ls_pred = z['ls_pred']
            y_ls = \
                self.encoder(
                    ops.nan_to_num(
                        ops.squeeze(
                            y[self.input_name_HR][:,
                                                  0,  # target lookback index
                                                  ...],
                            axis=1)))[0]  # use only the mean
            lspred_loss = self.loss_fn(z_ls_pred, y_ls) * self.alpha_ls
        else:
            lspred_loss = 0.0

        if 'KL' in self.loss_list:
            z_mean = z['mean']
            z_logvar = z['logvar']
            # kl loss variance formulation
            kl_loss = self.loss_fn_KL(z_mean, z_logvar, beta=self.beta)
        else:
            kl_loss = 0.0

        def y_k(k):
            return \
                y[self.input_name_HR][:,
                                      k,  # kth lookback index
                                      self.masking.rows,
                                      self.masking.cols,
                                      :]

        if 'reconstruction' in self.loss_list:
            z_ae_recons = z['ae_recons'][:,
                                         self.masking.rows,
                                         self.masking.cols,
                                         :]
            re_loss = self.loss_fn(z_ae_recons, y_k(1)) * self.gamma
        else:
            re_loss = 0.0

        if 'outer_pred' in self.loss_list:
            z_hybrid = z['hybrid'][:,
                                   self.masking.rows,
                                   self.masking.cols,
                                   :]
            # actual prediction
            pred_loss = self.loss_fn(z_hybrid, y_k(0)) * self.alpha
        else:
            pred_loss = 0.0

        # total loss
        loss = pred_loss + re_loss + lspred_loss + kl_loss

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
            if metric.name == "outer_pred":
                metric.update_state(pred_loss)
            if metric.name == "inner_pred":
                metric.update_state(lspred_loss)
            if metric.name == "recon":
                metric.update_state(re_loss)
            if metric.name == "KL":
                metric.update_state(kl_loss)

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

        assert resnet_result.shape[1:-1] == predictor_result.shape[1:-1], \
            "resnet and predictor have different rows/cols"

        if self.hybridization == 'product':
            # multiplicative skip connection
            assert resnet_result.shape == predictor_result.shape, \
                "resnet and predictor have different #filters"
            x = layers.Multiply()([resnet_result, predictor_result])
        elif self.hybridization == 'add':
            assert resnet_result.shape == predictor_result.shape, \
                "resnet and predictor have different #filters"
            x = layers.Add()([resnet_result, predictor_result])
        elif self.hybridization == 'concat':
            x = layers.Concatenate()(
                [resnet_result, predictor_result], axis=-1
            )

        # reusing resnet output block
        out = self.output_block(x)

        outputs = {'hybrid': out,
                   'mean': self.ae_mean,
                   'logvar': self.ae_logvar,
                   'ae_recons': self.ae_recons,
                   'ls_pred': self.ls_pred,
                   }

        inputs = {self.input_name_HR: input_HR,
                  self.input_name_LR: input_LR,
                  }

        return inputs, outputs
