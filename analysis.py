# compare runs
import plot_utils
import importlib
import dill
import numpy as np
import tools
import data_manager_cmems

import matplotlib.pyplot as plt
# plt.switch_backend('Agg')

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
    x = []
    y = []
    z = []
    t = []
    with open(fname, 'rb') as file:
        timeseries = dill.load(file)
        breakpoint()
        results = timeseries['results']
        truths = timeseries['truths']
        x = np.concatenate([re['LR_data'][:, 0,] for re in results], 0)
        z = np.concatenate([re['HR_data'][:, 0,] for re in results], 0)
        y = np.concatenate([tr['HR_data'][:, 0,] for tr in truths], 0)
        t = np.array([np.datetime64(re['time']) for re in results])

    return x, y, z, t


importlib.reload(plot_utils)
plot_machine = plot_utils.PlotMachine(dm=dmgr_cmems)
results_dir_org = plot_machine.results_dir
results_dir = results_dir_org + '/merge'
plot_machine.set_results_dir(results_dir)

# timeseries_hybrid = \
#     ('experiment/predictor_ESNcNr10e3Tikh5_v2/results/'
#      'timeseries.dill')

timeseries_resnet = \
    ('experiment/resnet_b6f64_bilin/results'
     '/results.dill')

timeseries_hybrid = \
    ('experiment/predictor_ESNcN10e3R1A1T0.5_6mpred/results/'
     'results.dill')

timeseries_hybrid2 = \
    ('experiment/predictor_ESNcN10e3R1A1T1_6mpred/results/'
     'results.dill')

timeseries_hybrid3 = \
    ('experiment/predictor_ESNcN10e3R1A1T0.1_6mpred/results/'
     'results.dill')

# timeseries_dmd = \
#     ('experiment/predictor_DMDcTikh5/results'
#      '/timeseries.dill')

x, y, z_resnet, t = load_timeseries(timeseries_resnet)
_, _, z_hybrid, _ = load_timeseries(timeseries_hybrid)
_, _, z_hybrid2, _ = load_timeseries(timeseries_hybrid2)
_, _, z_hybrid3, _ = load_timeseries(timeseries_hybrid3)
# _, _, z_dmd = load_timeseries(timeseries_dmd)

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
        'time': [],
        'plotkwargs': {
            'label': 'ESNc prediction T0.5',
            'linestyle': '-',
            'color': cmap(2),
            'zorder': 4,
        },
    },

    'pred_hybrid2': {
        'data': np.nan_to_num(z_hybrid2),
        'time': [],
        'plotkwargs': {
            'label': 'ESNc prediction T1',
            'linestyle': '--',
            'color': cmap(6),
            'zorder': 0,
        },
    },

    'pred_hybrid3': {
        'data': np.nan_to_num(z_hybrid3),
        'time': [],
        'plotkwargs': {
            'label': 'ESNc prediction T0.1',
            'linestyle': ':',
            'color': cmap(7),
            'zorder': 0,
        },
    },

    # 'pred_dmd': {
    #     'data': np.nan_to_num(z_dmd),
    #     'time': [],
    #     'plotkwargs': {
    #         'label': 'DMDc prediction',
    #         'linestyle': '-',
    #         'color': cmap(3),
    #         'zorder': 4,
    #     },
    # },
}

plt.switch_backend('qtagg')
plot_machine.plot_spectrum(data,
                           transect_name='along_flow',
                           spectrum_type='energy',
                           direction='temporal',
                           add_powerlaws=False)
plt.pause(1)
