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
                        .get_layer('hybrid_coupling')\
                        .output

        self.predictor_input_HR = \
            predictor_model.model.input['HR_data']
        self.predictor_input_LR = \
            predictor_model.model.input['LR_data']
        self.predictor_hidden = \
            predictor_model.model.input['hidden']

        self.predictor = predictor_model.predictor
        self.esn_dmd_pars = predictor_model.esn_dmd_pars

        self.predictor_output = \
            predictor_model.model.output['skip_vae_output']

        self.ae_mean = \
            predictor_model.model.output['mean']
        self.ae_hidden = \
            predictor_model.model.output['hidden']
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

        # use same encoder as predictor is using
        self.encoder = predictor_model.model.get_layer('encoder')

        self.predictor_layers = keras.Model(
            inputs={'HR_data': self.predictor_input_HR,
                    'LR_data': self.predictor_input_LR,
                    'hidden': self.predictor_hidden},
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

        self.loss_fn_KL = predictor_model.loss_KL

        self.trackers = []
        self.trackers.append(keras.metrics.Mean(name="loss"))

        for loss_name in self.loss_list:
            self.trackers.append(keras.metrics.Mean(name=loss_name))

    @property
    def metrics(self):
        return self.trackers

    def create_input(self, inputs):
        return {self.input_name_HR:
                ops.nan_to_num(inputs[self.input_name_HR]),
                self.input_name_LR:
                ops.nan_to_num(inputs[self.input_name_LR]),
                'hidden':
                inputs['hidden'],
                }

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
                            axis=1)),
                    training=training,
                )[0]  # use only the mean
            lspred_loss = self.loss_MSE(z_ls_pred, y_ls) * self.alpha_inner
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
            re_loss = self.loss_MSLE(z_ae_recons, y_k(1)) * self.gamma
        else:
            re_loss = 0.0

        if 'outer_pred' in self.loss_list:
            z_hybrid = z['hybrid'][:,
                                   self.masking.rows,
                                   self.masking.cols,
                                   :]
            # actual prediction
            pred_loss = self.loss_MSLE(z_hybrid, y_k(0)) * self.alpha_outer
        else:
            pred_loss = 0.0

        if 'ls_size' in self.loss_list:
            # latent space size loss
            z_mean = z['mean']
            ls_size = ops.mean(ops.abs(z_mean)) * self.alpha_ls
        else:
            ls_size = 0.0

        if False:
            ztest_hybrid = z['hybrid'][0, ..., 0].cpu().detach().numpy()
            ztest_resnet = z['resnet'][0, ..., 0].cpu().detach().numpy()
            ztest_recons = z['ae_recons'][0, ..., 0].cpu().detach().numpy()
            ztest_predictor = z['predictor'][0, ..., 0].cpu().detach().numpy()
            ztest_comb = z['combination'][0, ..., 0].cpu().detach().numpy()
            xtest = x['HR_data'][0, 0, ..., 0].cpu().detach().numpy()
            dtest = ztest_hybrid - xtest
            self.nanmask = ops.not_equal(self.masking.mask, 0.0)
            self.nanmask = \
                (self.masking.mask /
                 ops.cast(self.nanmask, self.masking.mask.dtype)
                 )
            self.nanmask = self.nanmask[..., 0].cpu().detach().numpy()
            import matplotlib.pyplot as plt
            plt.switch_backend('qtagg')
            plt.clf()
            plt.subplot(3, 3, 1)
            a = plt.pcolormesh(ztest_hybrid * self.nanmask)
            plt.colorbar(a)
            plt.gca().set_title('hybrid')
            plt.subplot(3, 3, 2)
            a = plt.pcolormesh(ztest_resnet * self.nanmask)
            plt.colorbar(a)
            plt.gca().set_title('resnet')
            plt.subplot(3, 3, 3)
            a = plt.pcolormesh(ztest_predictor * self.nanmask)
            plt.colorbar(a)
            plt.gca().set_title('predictor')
            plt.subplot(3, 3, 4)
            a = plt.pcolormesh(xtest)  # , vmin=0, vmax=1)
            plt.colorbar(a)
            plt.gca().set_title('truth')
            plt.subplot(3, 3, 5)
            a = plt.pcolormesh(ztest_comb * self.nanmask)
            plt.colorbar(a)
            plt.gca().set_title('combination')
            plt.subplot(3, 3, 6)
            a = plt.pcolormesh(ztest_recons * self.nanmask)
            plt.colorbar(a)
            plt.gca().set_title('reconstruction')
            plt.subplot(3, 3, 7)
            a = plt.pcolormesh(dtest)
            plt.colorbar(a)
            plt.gca().set_title('hybrid-truth')
            plt.pause(0.1)

        # denselr = self.model.\
        #     get_layer('encoder').\
        #     get_layer('vae_input_transform')

        # print(denselr.get_weights())
        # breakpoint()

        # total loss
        loss = pred_loss + re_loss + lspred_loss + kl_loss + ls_size

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
            if metric.name == "reconstruction":
                metric.update_state(re_loss)
            if metric.name == "ls_size":
                metric.update_state(ls_size)
            if metric.name == "KL":
                metric.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def builder(self):

        # reusing predictor input
        input_HR = self.predictor_input_HR
        input_LR = self.predictor_input_LR
        hidden = self.predictor_hidden

        assert input_HR.shape[1] > 1, \
            "need at least lookback=2 to make predictions"

        # reusing resnet input
        # input_LR = self.resnet_input

        resnet_result = \
            base_model.Activation('linear')(
                self.resnet_layers(input_LR))
        predictor_result = \
            base_model.Activation('linear')(
                self.predictor_layers({
                    'HR_data': input_HR,
                    'LR_data': input_LR,
                    'hidden': hidden,
                }))

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
            x = layers.Concatenate(axis=-1)(
                [resnet_result, predictor_result])
            x = layers.Conv2D(
                filters=resnet_result.shape[-1],
                kernel_size=3,
                strides=1,
                padding='same',
            )(x)
        elif self.hybridization == 'predictor':
            x = predictor_result
        elif self.hybridization == 'resnet':
            x = resnet_result
        else:
            raise Exception('invalid hybridization parameter')

        # reusing resnet output bplock
        # x = layers.BatchNormalization()(x)
        out = self.output_block(x)

        outputs = {'hybrid': out,
                   'resnet': resnet_result,
                   'predictor': predictor_result,
                   'combination': x,
                   'mean': self.ae_mean,
                   'hidden': self.ae_hidden,
                   'logvar': self.ae_logvar,
                   'ae_recons': self.ae_recons,
                   'ls_pred': self.ls_pred,

                   }

        inputs = {self.input_name_HR: input_HR,
                  self.input_name_LR: input_LR,
                  'hidden': hidden,
                  }

        return inputs, outputs
