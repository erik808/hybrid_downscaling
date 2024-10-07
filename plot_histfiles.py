import matplotlib.pyplot as plt
import dill
import plot_utils
from importlib import reload
reload(plot_utils)
from plot_utils import PlotMachine

files = {}

files['FT_inactive']=('experiments/gaussian_inactive_FT-default/'
                      'models/mdata_trial_1_20241005_204321.dill')

files['FT_only_2lrs']=('experiments/gaussian_FT_only_2lrs-default/'
                       'models/mdata_trial_0_20241005_232914.dill')

files['FT_hybrid_2lrs']=('experiments/gaussian_FT_hybrid_2lrs-default/'
                         'models/mdata_trial_0_20241005_232903.dill')


pm = PlotMachine()

plt.close('all')
plt.figure(figsize=(10,4), dpi=300)


for i,(key, item) in enumerate(files.items()):
    plot_baseline = False if i < len(files)-1 else True
    with open(item, 'rb') as file:
        data = dill.load(file)

    pm.plot_history(data['hist'], managed=True, add=key,
                    plot_baseline=plot_baseline)


plt.tight_layout()
plt.savefig('history.png')
    
        
        
