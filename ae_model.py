import keras
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

def build_model(conv_arch='7_conv_layers',
                learning_rate=0.001,
                use_dropout=True,
                activation='relu',
                mask=None):
    
    num_filters = 32
    num_filters_red = 32
    kernel_size = (3,3)

    masking_layer1 = Masking(mask, name="masking_layer1")
    masking_layer2 = Masking(mask, name="masking_layer2")
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
        x = layers.Dropout(0.25)(x)

    encoded = layers.Conv2D(num_filters_red, kernel_size, strides = (2,2),
                       activation=activation,
                       padding="same")(x)

    encoder = Model(state_input, encoded, name="encoder")
    encoder.summary(60)

    # Decoder ------------------------------------------------------
    y = layers.Conv2DTranspose(num_filters, kernel_size, strides=(2,2), activation=activation,
                               padding="same")(encoded)
    if conv_arch == '7_conv_layers':
        y = layers.Conv2DTranspose(num_filters, kernel_size, strides=(2,2), activation=activation,
                                   padding="same")(y)

    if use_dropout:
        y = layers.Dropout(0.25)(y)

    y = layers.Conv2DTranspose(num_filters, kernel_size, strides=(2,2), activation=activation,
                                padding="same")(y)
    y = layers.Conv2D(num_channels, kernel_size, activation="sigmoid",
                       padding="same")(y)

    if conv_arch == '7_conv_layers':
        cropped = layers.Cropping2D(cropping=((1,2),(3,4)))(y)
        
    elif conv_arch == '5_conv_layers':
        cropped = layers.Cropping2D(cropping=((2,1),(2,1)))(y)
    else:
        raise Exception(f'invalid conv_arch {conv_arch}')

    decoded = masking_layer2(cropped)

    decoder = Model(encoded, decoded, name="decoder")
    decoder.summary(60)

    autoencoder = Model(state_input, decoded, name="autoencoder")
    autoencoder.summary(60)

    loss = keras.losses.MeanSquaredError(
        reduction="sum_over_batch_size",
        name="mean_squared_error"
    )

    optim = keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optim,
                        loss=loss)

    return encoder, decoder, autoencoder
