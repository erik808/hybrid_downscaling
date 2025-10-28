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
            lookback: int = 4,
            shuffle: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dm = dm
        assert self.dm.ready, "data manager not ready"
        self.mode = mode
        self.batch_size = batch_size
        self.lookback = lookback
        assert lookback > 0, "lookback needs to be 1 or greater"
        self.shuffle = shuffle

        self.create_indices()

    def create_indices(self):
        self.index_range = self.dm.train_range if self.mode == 'train' \
            else self.dm.test_range

        # adjust to accommodate lookback
        self.index_range = \
            slice(np.max([self.index_range.start, self.lookback - 1]),
                  self.index_range.stop)

        # convert from slice to range in order to get a length
        self.indices = \
            np.arange(*self.index_range.indices(self.index_range.stop))
        self.n = len(self.indices)

        self.__do_shuffle()

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, index):
        low  = index * self.batch_size
        high = np.min([low + self.batch_size, self.n])
        inds = self.indices[low:high]
        lb_inds = np.stack([inds - i for i in range(self.lookback)], -1)
        
        batch_x = []
        batch_y = []
        # # implement lookback
        breakpoint()
        return (batch_x, batch_y)

    def __do_shuffle(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        self.__do_shuffle()


dmgr_cmems = data_manager_cmems.DataManagerCMEMS()
dmgr_cmems.create_training_data()

dgen_cmems = DataGeneratorCMEMS(dm=dmgr_cmems,
                                mode='train',
                                shuffle=False)

dgen_cmems.__getitem__(0)
