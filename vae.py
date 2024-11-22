import keras
from keras import ops
import torch


class VAE(keras.Model):

    def __init__(
            self,
            encoder,
            decoder,
            model='VAE',
            **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.model = model
        self.total_loss_tracker = \
            keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = \
            keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = \
            keras.metrics.Mean(name="KL_loss")
        self.rnn_loss_tracker = \
            keras.metrics.Mean(name="rnn_loss")


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
        z_mean = z[0]
        z_log_var = z[1]

        # print(ops.norm(z_mean), end="  ")
        # print(ops.norm(z_log_var))

        # Compute losses
        # reconstruction_loss = ops.mean(
        #         ops.sum(
        #             keras.losses.binary_crossentropy(y[0], y_pred),
        #             axis=(1, 2),
        #         )
        #     )

        RNN_mode = ('RNN' in self.model)
        if RNN_mode:
            tmp_input = x_state
            # only use this part anyway
            tmp_input[:,0,] = y[0]
            _, y_true = self.encoder(tmp_input)

            y_mean = y_true[2]
            y_log_var = y_true[3]

            rnn_loss = \
                ops.mean( ops.square(y_mean-z_mean) +
                          ops.square(y_log_var-z_log_var) )
            self.rnn_loss_tracker.update_state(rnn_loss)
            rnn_dict = {'rnn_loss' : self.rnn_loss_tracker.result()}
        else:
            rnn_loss = 0
            rnn_dict = {}


        reconstruction_loss = \
            ops.mean(ops.square(y[0]-y_pred))

        kl_loss = \
            -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))


        if RNN_mode:
            total_loss = reconstruction_loss + kl_loss + rnn_loss
        else:
            total_loss = reconstruction_loss + kl_loss


        total_loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        out_dict =  {
            'loss' : self.total_loss_tracker.result(),
            'reconstr_loss' : self.reconstruction_loss_tracker.result(),
            'KL_loss' : self.kl_loss_tracker.result()
        }
        if RNN_mode: out_dict.update(rnn_dict)
        return out_dict
