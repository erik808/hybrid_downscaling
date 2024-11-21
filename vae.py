import keras
import torch


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


    def call(self, inputs):
        enc_output, z_vars = self.encoder(inputs[0])
        pred = self.decoder([enc_output, inputs[1]])
        return pred


    def train_step(self, data):
        x, y = data

        if len(x) == 2:
            x_state, x_ft = x
        else:
            x_state = x
            
        self.zero_grad()

        # Forward pass
        enc_output, z = self.encoder(x_state)
        y_pred = self.decoder([enc_output, x_ft])

        # Compute loss
        loss = self.loss_fn(y[0], y_pred)

        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        self.total_loss_tracker.update_state(loss)

        return {
            'loss' : self.total_loss_tracker.result(),
            # 'reconstruction_loss' : self.reconstruction_loss_tracker.result(),
            # 'kl_loss' : self.kl_loss_tracker.result(),
            }
