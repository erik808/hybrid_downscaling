import sys

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

    def __init__(self, test_vec, mask, log_file):
        super(AutoEncoder, self).__init__()

        self.test_vec = test_vec
        self.mask = mask
        self.log_file = log_file
        self.dropout_rate=0.25

        self.log('AutoEncoder\n', 'w')

    def build_model(self,
                    conv_arch='7_conv_layers',
                    learning_rate=0.002,
                    use_dropout=False,
                    activation='relu',
                    optimizer='adam',
                    verbosity=0,
                    use_feedthrough=False,
                    ):

        Nlat, Nlon, num_channels = self.test_vec.shape
        num_filters = 32
        num_filters_red = 32
        kernel_size = (3,3)

        masking_layer = Masking(self.mask, name="masking_layer")
        masking_layer_ft = Masking(self.mask, name="masking_layer_ft")

        state_input = layers.Input(shape=(Nlat, Nlon, num_channels),
                                   name="full_state_input")

        # Encoder ------------------------------------------------------
        x = layers.Conv2D(num_filters, kernel_size, strides = (2,2),
                          activation=activation,
                          padding="same")(state_input)

        if conv_arch == '7_conv_layers':
            x = layers.Conv2D(num_filters, kernel_size, strides = (2,2),
                              activation=activation,
                              padding="same")(x)

        if use_dropout:
            x = layers.Dropout(self.dropout_rate)(x)

        encoded = layers.Conv2D(num_filters_red, kernel_size, strides = (2,2),
                           activation=activation,
                           padding="same")(x)

        encoder = Model(state_input, encoded, name="encoder")
        if verbosity > 10:
            encoder.summary(60)

        # Decoder ------------------------------------------------------
        y = layers.Conv2DTranspose(num_filters, kernel_size,
                                   strides=(2,2),
                                   activation=activation,
                                   padding="same")(encoded)
        if conv_arch == '7_conv_layers':
            y = layers.Conv2DTranspose(num_filters, kernel_size,
                                       strides=(2,2), activation=activation,
                                       padding="same")(y)

        if use_dropout:
            y = layers.Dropout(self.dropout_rate)(y)

        y = layers.Conv2DTranspose(num_filters, kernel_size,
                                   strides=(2,2), activation=activation,
                                    padding="same")(y)

        if not use_feedthrough:
            y = layers.Conv2D(num_channels, kernel_size, activation="sigmoid",
                              padding="same")(y)

        # Crop the decoded output
        if conv_arch == '7_conv_layers':
            cropped = layers.Cropping2D(cropping=((1,2),(3,4)))(y)

        elif conv_arch == '5_conv_layers':
            cropped = layers.Cropping2D(cropping=((2,1),(2,1)))(y)
        else:
            raise Exception(f'invalid conv_arch {conv_arch}')

        if use_feedthrough:
            feedthrough = layers.Input(shape=(Nlat, Nlon, num_channels),
                                       name="feedthrough_input")

            z = layers.Conv2D(num_filters, kernel_size,
                              strides = (1,1),
                              activation=activation,
                              padding="same")(feedthrough)

            output = layers.Concatenate()([decoded, z])
            output = layers.Conv2D(num_channels, kernel_size, activation="sigmoid",
                                   padding="same")(output)
            output = masking_layer_ft(output)

            inputs_decoder=[encoded, feedthrough]
            inputs_autoencoder=[state_input, feedthrough]
            outputs = [masking_layer(output)]

        else:
            inputs_decoder=[encoded]
            inputs_autoencoder=[state_input]
            outputs = [masking_layer(cropped)]

        decoder = Model(inputs=inputs_decoder,
                        outputs=outputs,
                        name="decoder")

        if verbosity > 10:
            decoder.summary(60)

        autoencoder = Model(inputs=inputs_autoencoder,
                            outputs=outputs,
                            name="autoencoder")

        if verbosity > 5:
            autoencoder.summary(60)

        loss = keras.losses.MeanSquaredError(
            reduction="sum_over_batch_size",
            name="mean_squared_error"
        )

        if optimizer == 'adam':
            optim = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optim = keras.optimizers.SGD(learning_rate=learning_rate)

        autoencoder.compile(optimizer=optim,
                            loss=loss)

        # write to log
        self.log(locals(), 'a')
        self.log_model(autoencoder, 'a')
        breakpoint()
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
