# compare runs
import plot_utils
import importlib
import dill
import data_manager_cmems

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

importlib.reload(data_manager_cmems)
importlib.reload(plot_utils)

# create datamanager
dmgr_cmems = \
    data_manager_cmems.DataManagerCMEMS(
        experiment_id='analysis',
        testing=False,
        force_rebuild=False,
        base_dir=".",
    )

plot_machine = plot_utils.PlotMachine(dm=dmgr_cmems)


def load_timeseries(fname):
    with open(fname, 'rb') as file:
        timeseries = dill.load(file)
        x = timeseries['x']
        y = timeseries['y']
        z = timeseries['z']

    return x, y, z


timeseries_hybrid = \
    ('experiments/hybrid_dmdcL1e0_vae5lf64-256_det/'
     'results/epoch5_20251202_151148/timeseries.dill')

timeseries_resnet = \
    ('experiments/resnetb6f64o0/results'
     '/epoch12_20251202_152349/timeseries.dill')

results_dir_org = plot_machine.results_dir

x, y, z = load_timeseries(timeseries_resnet)
results_dir = results_dir_org + '/resnet'
plot_machine.set_results_dir(results_dir)
plot_machine.spectra_wrapper(x, y, z)


x, y, z = load_timeseries(timeseries_hybrid)
results_dir = results_dir_org + '/hybrid'
plot_machine.set_results_dir(results_dir)
plot_machine.spectra_wrapper(x, y, z)
