import numpy as np
import keras
from data_manager_cmes import DataManagerCMEMS


class DataFactory():
    def __new__(
            cls,
            testing_mode=False,
            case_study='cmems',
    ):
        if case_study == 'cmems':
            return DataManagerCMEMS(testing_mode)
        else:
            raise ValueError("unknown case study")


class DataGenerator(keras.utils.PyDataset):

    def __init__(self, x, y,
                 ft_type='hybrid',
                 batch_size=4,
                 shuffle=False,
                 lookback=0,
                 encoder=None,
                 unroll_dim=1,
                 **kwargs):

        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.ft_type = ft_type
        self.shuffle = shuffle
        self.lookback = lookback
        self.encoder = encoder
        self.unroll_dim = unroll_dim

        self.unroll_y = (self.unroll_dim > 0 and
                         self.ft_type == 'hybrid')

        self.unroll_x = self.unroll_y

        self.__setup_data(x, y)

    def __setup_data(self, x, y):
        assert len(x) == 2
        assert len(y) == 1

        self.indices = np.arange(self.lookback,
                                 x[0].shape[0] - self.unroll_dim)

        self.n = len(self.indices)
        self.__do_shuffle()

        if self.ft_type == 'hybrid':
            self.x = x
        elif self.ft_type == 'only':
            self.x = [x[1]]
        else:
            self.x = [x[0]]

        self.y = y

        if self.unroll_x:
            # set of shifted feedthrough inputs. self.indices is
            # truncated with unroll_dim to make sure this does not
            # lead to problems.
            ft_set = [x[1][i:,] for i in range(self.unroll_dim + 1)]
            # append feedthroughs to state input
            self.x =[x[0]] + ft_set

        if self.unroll_y:
            self.y = [y[0][i:,] for i in range(self.unroll_dim + 1)]

    def __len__(self):
        # number of batches
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, index):
        low  = index * self.batch_size
        high = np.min([low + self.batch_size, self.n])
        inds = self.indices[low:high]
        batch_x = create_lookback(inds, self.x, self.lookback)
        batch_y = create_lookback(inds, self.y, self.lookback)
        # batch_y = [y[inds,] for y in self.y]
        return (batch_x, batch_y)

    def __do_shuffle(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        self.__do_shuffle()


def create_lookback(inds, data, lookback, axis=1):
    batch = list()

    for var in data:
        lb_fields = list()  # lookback fields
        for lb in range(lookback + 1):
            lb_field = var[inds - lb,]
            lb_fields.append(lb_field)

        batch.append(np.stack(lb_fields, axis=axis))

    return batch


class CustomScaler():

    def __init__(self, scaling_type='minmax_per_feature'):
        self.scaling_type = scaling_type
        self.shift = None
        self.scale = None
        self.fitted = False

    def fit(self, data):
        if self.scaling_type == 'standardize_per_feature':
            self.shift = np.mean(data, axis=0)
            self.scale = np.mean(data, axis=0)

        elif self.scaling_type == 'standardize_over_all_features':
            self.shift = np.mean(data)
            self.scale = np.mean(data)

        elif self.scaling_type == 'minmax_per_feature':
            self.scale = 1.0 / (np.max(data, axis=0) - np.min(data, axis=0))
            self.shift = np.min(data, axis=0)

        elif self.scaling_type == 'minmax_over_all_features':
            self.scale = 1.0 / (np.max(data) - np.min(data))
            self.shift = np.min(data)

        self.fitted = True

    def transform(self, data):
        if not self.fitted:
            raise Exception('scaler not fitted')
        return (data - self.shift) * self.scale

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        if not self.fitted:
            raise Exception('scaler not fitted')
        return (data / self.scale) + self.shift
