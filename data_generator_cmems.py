import data_manager_cmems
import importlib
import keras

importlib.reload(data_manager_cmems)


class DataGeneratorCMEMS(keras.utils.PyDataset)

dmgr_cmems = data_manager_cmems.DataManagerCMEMS()
dmgr_cmems.create_training_data()
