# compare runs
import plot_utils
import importlib
import dill
import numpy as np
import tools
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


def load_timeseries(fname):
    with open(fname, 'rb') as file:
        timeseries = dill.load(file)
        x = timeseries['x']
        y = timeseries['y']
        z = timeseries['z']

    return x, y, z


importlib.reload(plot_utils)
plot_machine = plot_utils.PlotMachine(dm=dmgr_cmems)
results_dir_org = plot_machine.results_dir
results_dir = results_dir_org + '/merge'
plot_machine.set_results_dir(results_dir)

# timeseries_hybrid = \
#     ('experiment/train_predictor/results/'
#      'timeseries.dill')

# timeseries_hybrid = \
#     ('experiment/predictor_ESNcNr10e3Tikh5_v2/results/'
#      'timeseries.dill')

timeseries_hybrid = \
    ('experiment/predictor_ESNcTikh5_earlystop/results/'
     'timeseries.dill')


# predictor_ESNcNr10e3Tikh5_v2
# ('experiment/hybrid_dmdcL1e-6cut0.1/results/epoch9_20251203_103824/'
#  'timeseries.dill')
# 'experiment/hybrid_dmdcL1e-7cut0.1/results/timeseries.dill'

timeseries_resnet = \
    ('experiment/resnetb6f64o0/results/epoch59_20251202_191553'
     '/timeseries.dill')

timeseries_dmd = \
    ('experiment/predictor_DMDcTikh5/results'
     '/timeseries.dill')

x, y, z_resnet = load_timeseries(timeseries_resnet)
_, _, z_hybrid = load_timeseries(timeseries_hybrid)
_, _, z_dmd = load_timeseries(timeseries_dmd)

scaler_list = ['LR', 'HR', 'HR', 'HR']
x, y, z_resnet, z_hybrid = \
    [tools.unscale_var(d, dmgr_cmems.scalers[res])
     for d, res in zip([x, y, z_resnet, z_hybrid], scaler_list)]


if x.shape != y.shape:
    # upsample unscaled x (bilinear interpolation)
    x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))
    x = dmgr_cmems\
        .bilin_upsampler(x)\
        .transpose((0, 2, 3, 1))

plt.close('all')

cmap = plt.get_cmap('tab10')
data = {
    'truth': {
        'data': np.nan_to_num(y),
        'time': [],
        'plotkwargs': {
            'label': 'high-resolution truth',
            'linestyle': '-',
            'color': cmap(0),
            'zorder': 10,
        },
    },

    # 'lowres': {
    #     'data': np.nan_to_num(x),
    #     'time': [],
    #     'plotkwargs': {
    #         'label': 'bilinear interpolation',
    #         'linestyle': '-',
    #         'color': cmap(5),
    #         'zorder': 0,
    #     },
    # },

    'pred_resnet': {
        'data': np.nan_to_num(z_resnet),
        'time': [],
        'plotkwargs': {
            'label': 'SRResNet prediction',
            'linestyle': '-',
            'color': cmap(1),
            'zorder': 5,
        },
    },

    'pred_hybrid': {
        'data': np.nan_to_num(z_hybrid),
        'plotkwargs': {
            'label': 'ESNc prediction',
            'linestyle': '-',
            'color': cmap(2),
            'zorder': 4,
        },
    },
    
    'pred_dmd': {
        'data': np.nan_to_num(z_dmd),
        'time': [],
        'plotkwargs': {
            'label': 'DMDc prediction',
            'linestyle': '-',
            'color': cmap(3),
            'zorder': 4,
        },
    },
}

plt.switch_backend('qtagg')
plot_machine.plot_spectrum(data,
                           transect_name='along_flow',
                           spectrum_type='ssh',
                           direction='temporal',
                           add_powerlaws=False)
plt.pause(1)
