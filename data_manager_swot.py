import tools
import xarray as xr
from data_manager_base import DataManagerBase
from sklearn.preprocessing import MinMaxScaler


class DataManagerSWOT(DataManagerBase):

    def __init__(
            self,
            testing_mode=False
    ):
        tools.load_config(self, config_name='data_config_swot')
        self.data_dir = 'data'
        self.swot_duacs_fname = (f'{self.data_dir}/'
                                 'subset_merge_swot_duacs_1y.nc')

    def create_training_data(self):

        self.ds = xr.open_dataset(self.swot_duacs_fname)

        data_HR = self.ds.ssha.data
        data_LR = self.ds.sla.data

        Nt, Nlat, Nlon = data_HR.shape
        assert data_LR.shape == data_HR.shape

        scaler = MinMaxScaler(feature_range=self.scaling_range)
        data_HR = scaler.fit_transform(data_HR.reshape(Nt, -1))\
                        .reshape(Nt, Nlat, Nlon, 1)
        data_LR = scaler.transform(data_LR.reshape(Nt, -1))\
                        .reshape(Nt, Nlat, Nlon, 1)

        import matplotlib.pyplot as plt
        plt.close('all')
        plt.imshow(data_LR[50, :, :,])
        plt.pause(1)

        breakpoint()
