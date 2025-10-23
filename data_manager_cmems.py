import tools
from data_manager_base import DataManagerBase


class DataManagerCMEMS(DataManagerBase):

    def __init__(self):
        super().__init__()
        tools.load_config(self, config_name='data_config_cmems')

    def create_training_data(self):
        pass

    def load_uv_data(self):        
        print(self.uv_data_files)














    

dmgr_cmems = DataManagerCMEMS()
print(dir(dmgr_cmems))

dmgr_cmems.coords_file
