import dask.array as da
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
            lookback: int = 1,
            shuffle: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dm = dm
        assert self.dm.ready, "data manager not ready"

        self.mode = mode
        assert mode == 'train' or mode == 'test', "invalid mode"

        self.batch_size = batch_size
        self.lookback = lookback
        assert lookback > 0, "lookback needs to be greater than 0"

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

        HR_data, _ = \
            self.create_batch(inds, self.dm.ds_HR,
                              self.lookback, self.dm.scalers['HR'])
        LR_data, time = \
            self.create_batch(inds, self.dm.ds_LR,
                              self.lookback, self.dm.scalers['LR'])

        batch_x = {'LR_data': LR_data,
                   'HR_data': HR_data,
                   'meta' : {
                       # 'time':
                       # time.astype("datetime64[s]").astype(np.float32),
                       'mask': self.dm.mask.values,
                       'grid_HR': self.dm.grid_HR,
                       'grid_LR': self.dm.grid_LR,
                       # 'vars': list(self.dm.ds_HR.data_vars),
                   },
                   }

        batch_y = {'HR_data': HR_data}

        return (batch_x, batch_y)

    def create_batch(self, inds, ds, lookback, scaler, axis=1):

        # create indices including lookback
        lb_inds = np.stack([inds - i for i in range(lookback)], -1)

        # sort unique indices
        lb_inds_unique = np.sort(np.unique(lb_inds.flatten()))

        # indices mapped to restriction
        lb_inds_mapped = np.searchsorted(lb_inds_unique, lb_inds)

        # select subset and create data array
        darr = ds.isel(time=lb_inds_unique).load().to_array()

        # correct ordering
        darr = darr.transpose('time',
                              'latitude',
                              'longitude',
                              'variable').data

        # apply scaling
        darr_shape = darr.shape
        darr = darr.reshape(darr_shape[0], -1)
        darr = scaler.transform(darr).reshape(darr_shape)

        # testing -> move this to a unittest
        # test = da.zeros_like(da_HR)
        # for i in range(test.shape[0]):
        #     test[i,] = i

        # test_stacked = da.stack(
        #     [test[lb_inds_mapped[..., i],]
        #      for i in range(self.lookback)],
        #     axis=1)

        # stack the lookback in axis=1, lookback is backward in time
        darr_stacked = da.stack(
            [darr[lb_inds_mapped[..., i],]
             for i in range(lookback)],
            axis=1).compute()

        # get a corresponding time array
        time = ds.time[lb_inds.flatten()].data.reshape(lb_inds.shape)

        return darr_stacked, time

    def __do_shuffle(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        self.__do_shuffle()


def getter(**args):
    dgen_train = DataGeneratorCMEMS(mode='train', **args)
    args.update({'shuffle': False})
    print(f"(dgen_test): setting shuffle to {args['shuffle']}")
    dgen_test = DataGeneratorCMEMS(mode='test', **args)
    return dgen_train, dgen_test
