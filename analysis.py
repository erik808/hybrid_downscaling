# compare runs
import plot_utils
import importlib
import dill
import data_manager_cmems

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


timeseries1 = \
    ('experiments/hybrid_dmdcL1e0/results/'
     'epoch5_20251202_145818/timeseries.dill')
with open(timeseries1, 'rb') as file:
    timeseries = dill.load(file)
    x = timeseries['x']
    y = timeseries['y']
    z = timeseries['z']

plot_machine.spectra_wrapper(x, y, z)
