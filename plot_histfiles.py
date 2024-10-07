import matplotlib.pyplot as plt
import dill

files = {}

files['FT_inactive']=('experiments/gaussian_inactive_FT-default/'
                      'models/mdata_trial_1_20241005_204321.dill')

files['FT_only_2lrs']=('experiments/gaussian_FT_only_2lrs-default/'
                       'models/mdata_trial_0_20241005_232914.dill')

files['FT_hybrid_2lrs']=('experiments/gaussian_FT_hybrid_2lrs-default/'
                         'models/mdata_trial_0_20241005_232903.dill')


for key, item in files.items():

    with open(item, 'rb') as file:
        data = dill.load(file)


        
