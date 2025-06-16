import tools
from data_manager_base import DataManagerBase


class DataManagerSWOT(DataManagerBase):

    def __init__(
            self,
            testing_mode=False
    ):
        tools.load_config(self, config_name='data_config_swot')
        breakpoint()

    def create_training_data(self):
        pass
