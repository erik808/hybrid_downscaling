import tools
from data_manager_base import DataManagerBase


class DataManagerCMEMS(DataManagerBase):

    def __init__(self):
        super().__init__()
        tools.load_config(self, config_name='data_config_cmems')

    def create_training_data(self):
        pass


dmgr_cmems = DataManagerCMEMS()
print(dir(dmgr_cmems))
