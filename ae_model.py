import sys
import numpy as np

import keras
import keras_tuner
from keras import layers
from keras import ops
from keras.models import Model

# create custom masking class
@keras.saving.register_keras_serializable(name="custom_masking")
class Masking(layers.Layer):
    def __init__(self, mask, **kwargs):
        super(Masking, self).__init__(**kwargs)
        self.mask = mask

    def get_config(self):
        config = super(Masking, self).get_config()
        config.update({
            'mask' : keras.saving.serialize_keras_object(self.mask)})
        return config

    @classmethod
    def from_config(cls, config):
        mask_config = config.pop("mask")
        mask = keras.saving.deserialize_keras_object(mask_config)
        return cls(mask, **config)

    def call(self, inputs):
        return ops.multiply(inputs, self.mask)

class AutoEncoder(keras_tuner.HyperModel):

    def __init__(self, test_vec, mask, log_file, esn=None):
        super(AutoEncoder, self).__init__()

        self.test_vec = test_vec
        self.mask = mask
        self.log_file = log_file
        self.dropout_rate = 0.25
        self.esn = esn
        self.log('AutoEncoder\n', 'w')

        self.num_filters = 32
        self.num_filters_red = 16
        self.kernel_size = (3,3)
        self.num_resblocks = 2
        self.resblock_ctr = 0

    def ResBlock(self, inputs):
        self.resblock_ctr += 1
        x = layers.Conv2D(self.num_filters,
                          self.kernel_size,
                          padding="same",
                          activation="relu",
                          name=f"residual_block_conv2d_a_{self.resblock_ctr}")(inputs)
        x = layers.Conv2D(self.num_filters,
                          self.kernel_size,
                          padding="same",
                          name=f"residual_block_conv2d_b_{self.resblock_ctr}")(x)
        x = layers.Add(name=f"residal_block_add_{self.resblock_ctr}")([inputs, x])
        return x

    def build_model(self,
                    conv_arch='default',
                    learning_rate=0.002,
                    use_dropout=False,
                    activation='relu',
                    optimizer='adam',
                    verbosity=20,
                    use_feedthrough=False,                    
                    feedthrough_only=False,
                    use_timeinput=True,
                    feedthrough_type='multiply'
                    ):

        self.use_feedthrough = use_feedthrough        
        self.feedthrough_only = feedthrough_only
        self.feedthrough_type = feedthrough_type
        if self.feedthrough_only: self.use_feedthrough = True

        self.use_timeinput = use_timeinput
        self.use_dropout = use_dropout

        Nlat, Nlon, num_channels = self.test_vec.shape

        masking_layer = Masking(self.mask, name="masking_layer")
        masking_layer_ft = Masking(self.mask, name="masking_layer_ft")

        state_input = layers.Input(shape=(Nlat, Nlon, num_channels),
                                   name="full_state_input")

        if self.use_timeinput:
            time_input = layers.Input(shape=(1,1,1),
                                      name="time_input")

        if self.use_feedthrough:
            feedthrough = layers.Input(shape=(Nlat, Nlon, num_channels),
                                       name="feedthrough_input")

        # Encoder ------------------------------------------------------
        conv_layer_0 = layers.Conv2D(self.num_filters_red,
                                     self.kernel_size,
                                     strides = (1,1),
                                     activation=activation,
                                     padding="same", name="conv_layer_0")

        conv_layer_1 = layers.Conv2D(self.num_filters,
                                     self.kernel_size,
                                     strides = (2,2),
                                     activation=activation,
                                     padding="same", name="conv_layer_1")

        conv_layer_2 = layers.Conv2D(self.num_filters,
                                     self.kernel_size,
                                     strides = (2,2),
                                     activation=activation,
                                     padding="same", name="conv_layer_2")

        conv_layer_3 = layers.Conv2D(self.num_filters_red,
                                     self.kernel_size,
                                     strides = (2,2),
                                     activation=activation,
                                     padding="same", name="conv_layer_3")

        dropout_layer_1 = layers.Dropout(self.dropout_rate,
                                         name="dropout_1")

        x = conv_layer_1(state_input)
        x = conv_layer_2(x)
        if use_dropout:
            x = dropout_layer_1(x)
        encoded = conv_layer_3(x)

        encoder = Model([state_input, time_input], encoded, name="encoder")

        # Call ESN layer in the latent space
        if (self.esn != None and
            self.use_feedthrough):

            c = conv_layer_1(feedthrough)
            c = conv_layer_2(c)
            if use_dropout:
                c = dropout_layer(c)
            control = conv_layer_3(c)

            encoded = self.esn(encoded, time_input, control)

        # Decoder layers
        convt_layer_1 = layers.Conv2DTranspose(self.num_filters,
                                               self.kernel_size,
                                               strides=(2,2),
                                               activation=activation,
                                               padding="same",
                                               name='convt_layer_1')

        convt_layer_2 = layers.Conv2DTranspose(self.num_filters,
                                               self.kernel_size,
                                               strides=(2,2),
                                               activation=activation,
                                               padding="same",
                                               name='convt_layer_2')

        convt_layer_2_1 = layers.Conv2DTranspose(self.num_filters,
                                                 self.kernel_size,
                                                 strides=(1,1),
                                                 activation=activation,
                                                 padding="same",
                                                 name='convt_layer_2_1')

        convt_layer_3 = layers.Conv2DTranspose(self.num_filters,
                                               self.kernel_size,
                                               strides=(2,2),
                                               activation=activation,
                                               padding="same",
                                               name='convt_layer_3')

        dropout_layer_2 = layers.Dropout(self.dropout_rate,
                                         name="dropout_2")

        cropping_layer = layers.Cropping2D(cropping=((1,2),(3,4)),
                                           name='cropping_layer')

        feedthrough_layer_1 = layers.Conv2D(self.num_filters,
                                            self.kernel_size,
                                            strides = (1,1),
                                            activation=activation,
                                            padding="same",
                                            name='feedthrough_layer_1')

        output_layer = layers.Conv2D(num_channels, self.kernel_size,
                                     activation="sigmoid",
                                     padding="same",
                                     name='output_layer')

        # Decoder:
        y = convt_layer_1(encoded)
        for i in range(self.num_resblocks):
            y = self.ResBlock(y)
        y = convt_layer_2(y)
        # y = convt_layer_2_1(y)
        if use_dropout:  y = dropout_layer_2(y)
        y = convt_layer_3(y)
        cropped_y = cropping_layer(y)

        if self.feedthrough_only:
            output = feedthrough_layer_1(feedthrough)
            output = output_layer(output)

            inputs_decoder=[feedthrough]
            inputs_autoencoder=[feedthrough]
            outputs = [masking_layer(output)]

        elif self.use_feedthrough:
            z = feedthrough_layer_1(feedthrough)

            if feedthrough_type == 'concatenate':
                output = layers.Concatenate()([cropped_y, z])
            elif feedthrough_type == 'multiply':
                output = layers.Multiply()([cropped_y, z])
            elif feedthrough_type == 'ignore':
                output = cropped_y
            else:
                raise Exception('specify feedthrough_type when using feedthrough')

            output = output_layer(output)
            inputs_decoder=[encoded, feedthrough]
            inputs_autoencoder=[state_input, time_input, feedthrough]
            outputs = [masking_layer(output)]

        else:
            output = output_layer(cropped_y)
            inputs_decoder=[encoded]
            inputs_autoencoder=[state_input, time_input]
            outputs = [masking_layer(output)]


        # Construct models
        decoder = Model(inputs=inputs_decoder,
                        outputs=outputs,
                        name="decoder")

        autoencoder = Model(inputs=inputs_autoencoder,
                            outputs=outputs,
                            name="autoencoder")

        loss = keras.losses.\
            MeanSquaredError(reduction="sum_over_batch_size",
                             name="mean_squared_error")

        if optimizer == 'adam':
            optim = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optim = keras.optimizers.SGD(learning_rate=learning_rate)

        autoencoder.compile(optimizer=optim, loss=loss)

        self.log(locals(), 'a')
        self.log_model(autoencoder, 'a')

        return autoencoder, encoder, decoder

    # build model for hyperparameter tuning
    def build(self, hp):
        learning_rate = hp.Float("learning_rate",
                                 min_value=1e-4,
                                 max_value=5e-2,
                                 sampling="log")
        hypermodel, _, _ = self.build_model(learning_rate=learning_rate)
        return hypermodel

    def fit(self, hp, model, *args, **kwargs):
        batch_sizes = hp.Int("batch_size",
                             min_value=2,
                             max_value=32)
        return model.fit(*args, batch_size=batch_sizes,
                         **kwargs)

    def log(self, msg, mode='a'):
        original = sys.stdout
        with open(self.log_file, mode) as f:
            sys.stdout = f
            print(msg)
            sys.stdout = original

    def log_model(self, model, mode='a'):
        original = sys.stdout
        with open(self.log_file, mode) as f:
            sys.stdout = f
            print(model.summary())
            sys.stdout = original


class TriggerESN(keras.callbacks.Callback):
    """Callback to control the ESN during training of the AE

    This is very flexible but for now we just want to trigger training
    at the beginning of an epoch and train every x epochs

    Either select a stride <train_every> or train in selected epochs
    <train_in_epochs>.

    """

    def __init__(self, esn, train_every=1,
                 train_in_epochs=[],
                 num_samples=0):
        super().__init__()
        self.esn = esn
        self.train_every = train_every
        self.train_in_epochs = train_in_epochs
        self.num_samples = num_samples

    def on_epoch_begin(self, epoch, logs=None):

        # synchronize the size of the training data between AE and ESN
        # not sure if this is the right place
        self.esn.num_samples = self.num_samples

        if len(self.train_in_epochs) > 0:
            if epoch in self.train_in_epochs:
                self.esn.esn_ready_to_train[1] = True
        elif not epoch % self.train_every:
            self.esn.esn_ready_to_train[1] = True


class CustomValidation(keras.callbacks.Callback):
    """
    """

    def __init__(self, test_data, initial_xk, plotmachine, pars):
        super().__init__()

        self.initial_xk = initial_xk
        self.test_data = test_data[0]
        self.N_steps = self.test_data.shape[0]
        self.T_test = test_data[1]
        self.test_data_ft = test_data[2]
        self.plotmachine = plotmachine
        self.pars = pars

    def on_epoch_end(self, epoch, logs=None):
        predictions = np.zeros_like(self.test_data)

        xk   = self.initial_xk[0]
        xkm1 = self.initial_xk[1]
        pb_i = keras.utils.Progbar(self.N_steps,
                                   stateful_metrics=['error', 'base'])

        error, base = (0,0)
        for i in range(self.N_steps):

            xk_LR = np.expand_dims(self.test_data_ft[i,], axis=0)
            if self.pars['residual_mode']:
                # secant predictor in residual mode
                Pxk = 2*xk - xkm1 - xk_LR
            else:                    
                Pxk = xk_LR
                
            tid = np.expand_dims(self.T_test[i,], axis=0)
            xkm1 = xk

            if self.pars['feedthrough_only']:
                xk = self.model.predict([Pxk], verbose=0)
            elif self.pars['use_feedthrough']:
                xk = self.model.predict([xk, tid, Pxk], verbose=0)
            else:
                xk = self.model.predict([xk, tid], verbose=0)

            pred = xk + xk_LR if residual_mode else xk
            predictions[i,] = pred

            breakpoint()
            # CHECK SANITY!!
            
            xk_true = np.expand_dims(self.test_data[i,], axis=0)
            error += (np.sum(np.square(pred - xk_true)))
            base += (np.sum(np.square(xk_LR - xk_true)))
            values = [('error', np.sqrt(error/(i+1))),
                      ('base', np.sqrt(base/(i+1)))]
            pb_i.add(1, values=values)            
        

        self.plotmachine.plot_prediction_error(self.test_data,
                                               predictions,
                                               self.test_data_ft,
                                               f'_epoch_{epoch}')
        logs['val_error']=np.sqrt(error/(i+1))
        logs['val_base']=np.sqrt(base/(i+1))

        return predictions
