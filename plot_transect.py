import matplotlib.pyplot as plt
import dill
import plot_utils
from importlib import reload
reload(plot_utils)
from plot_utils import PlotMachine

import compute_tool
reload(compute_tool)
from compute_tool import ComputeTool
import data_manager
from data_manager import DataFactory


dm = DataFactory()
data, params, scalers, _ = \
    dm.create_training_data(compute_data=False,
                            detide=False,
                            coarsening_method='gaussian_filter',
                            sigma=[1,1.5,1.5],
                            truncation=100)

ct=ComputeTool()

snapshot = 5000

vort = ct.vorticity(data['train']['HR'][snapshot,], scalers['HR'])

plt.close('all')
plt.figure(figsize=(13,10),dpi=300)
h = plt.pcolormesh(vort,vmin=-10,vmax=10,cmap='RdBu')
plt.colorbar(h, label='vorticity (cycles/day)')

transect_names = ['along_flow', 'across_flow']
for transect_name in transect_names:
    dill_file = f'{dm.transect_dir}/{transect_name}.dill'
    print(f'Loading transect from {dill_file}')
    with open(dill_file, 'rb') as file:
        tpicker = dill.load(file)['tpicker']

    # -1 because of the padding in vorticity plot
    xidstart = tpicker.x_trans[0]-1
    xidstop = tpicker.x_trans[-1]-1
    yidstart = tpicker.y_trans[0]-1
    yidstop = tpicker.y_trans[-1]-1
    plt.plot([xidstart,xidstop],
             [yidstart,yidstop],
             'k--',linewidth=4,alpha=0.7)

plt.tight_layout()
plt.savefig('transects.png')
