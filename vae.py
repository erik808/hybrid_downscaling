import keras


class VAE(keras.Model):

    def __init__(
            self,
            encoder,
            decoder,
            **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = \
            keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = \
            keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = \
            keras.metrics.Mean(name="kl_loss")
        self.loss_fn = \
            keras.losses.MeanSquaredError()


    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]



    def train_step(self, data):
        x, y = data

        if len(x) == 2:
            x_state, x_ft = x
        else:
            x_state = x

        breakpoint()

        self.zero_grad()

        # Forward pass
        enc_output, z = self.encoder(x_state)
        y_pred = self.decoder(z, x_ft)

        # Compute loss
        loss = loss_fn(y, y_pred)

        loss.backward()
                
