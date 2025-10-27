import data_manager_cmems
import importlib
import keras

importlib.reload(data_manager_cmems)


class DataGeneratorCMEMS(keras.utils.PyDataset):
    def __init__(
            self,
            dm,
            batch_size: int = 4,
            lookback: int = 0,
            shuffle: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dm = dm
        self.batch_size = batch_size
        self.lookback = lookback
        self.shuffle = shuffle

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def __do_shuffle(self):
        pass

    def on_epoch_end(self):
        self.__do_shuffle()


dmgr_cmems = data_manager_cmems.DataManagerCMEMS()
dmgr_cmems.create_training_data()

dgen_cmems = DataGeneratorCMEMS(dm=dmgr_cmems)
