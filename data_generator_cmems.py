import data_manager_cmems
import importlib
import keras
import numpy as np

importlib.reload(data_manager_cmems)


class DataGeneratorCMEMS(keras.utils.PyDataset):
    def __init__(
            self,
            dm,
            mode: str = 'train',  # 'train' or 'test'
            batch_size: int = 4,
            lookback: int = 0,
            shuffle: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dm = dm
        assert self.dm.ready, "data manager not ready"
        self.batch_size = batch_size
        self.lookback = lookback
        self.shuffle = shuffle
        self.index_range = self.dm.train_range if mode == 'train' \
            else self.dm.test_range

        # adjust to accommodate lookback
        self.index_range = \
            slice(np.max([self.index_range.start, self.lookback]),
                  self.index_range.stop)

        # convert from slice to range in order to get a length
        self.n = \
            len(range(*self.index_range.indices(self.index_range.stop)))

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, index):
        low  = index * self.batch_size
        high = np.min([low + self.batch_size, self.n])
        inds = self.indices[low:high]
        # # implement lookback
        breakpoint()

    def __do_shuffle(self):
        pass

    def on_epoch_end(self):
        self.__do_shuffle()


dmgr_cmems = data_manager_cmems.DataManagerCMEMS()
dmgr_cmems.create_training_data()

dgen_cmems = DataGeneratorCMEMS(dm=dmgr_cmems,
                                mode='train')
