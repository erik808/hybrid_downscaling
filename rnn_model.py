import importlib
import keras
import tools
import base_model
import resnet_model
import vae_model

importlib.reload(base_model)
importlib.reload(resnet_model)
importlib.reload(vae_model)


class RNNmodel(base_model.BaseModel):
    def __init__(
            self,
            vae_model,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.vae_model = vae_model
        # disable training on VAE model
        self.vae_model.trainable = False

        breakpoint()

        tools.load_config(self, config_name='rnn_model')

        self.compiler = keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.re_loss_tracker = keras.metrics.Mean(name="recons")
