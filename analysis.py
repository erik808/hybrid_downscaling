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
    x, y, z, t = [], [], [], []
    with open(fname, 'rb') as file:
        timeseries = dill.load(file)
        results = timeseries['results'] if 'results' in timeseries else []
        truths = timeseries['truths'] if 'truths' in timeseries else []

        if len(truths) == 0 and not isinstance(results, list):
            z = results.cpu().detach().numpy()
        else:
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

timeseries_reference = \
    ('experiment/resnet_b6f64_bilin/results'
     '/results.dill')

hybrid_bases = [
    'experiment/predictor_ESNcT5e-3_6mpred_ks6000/results/',
    # 'experiment/predictor_ESNcT1e-2_6mpred_ks5000/results/',
    'experiment/predictor_ESNcT1e-2_6mpred_ks6000/results',
    'experiment/predictor_ESNcT5e-2_6mpred_ks6000/results',
    # 'experiment/predictor_ESNcT5e-3_6mpred_ks6000/results/'
]

timeseries_hybrid = \
    [(f'{hybrid_base}/results.dill') for hybrid_base in hybrid_bases]

plt.close('all')

x, y, z_resnet, t = load_timeseries(timeseries_reference)
z_hybrid = [load_timeseries(ts)[2] for ts in timeseries_hybrid]

scaler_list = ['LR', *('HR ' * (len(z_hybrid)+2)).split(' ')[:-1]]

# x, y, z_resnet, z_hybrid
fields = [x, y, z_resnet, *z_hybrid]
out = \
    [tools.unscale_var(d, dmgr_cmems.scalers[res])
     for d, res in zip(fields, scaler_list)]

x = out[0]
y = out[1]
z_resnet = out[2]
z_hybrid = out[3:]

if x.shape != y.shape:
    # upsample unscaled x (bilinear interpolation)
    x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))
    x = dmgr_cmems\
        .bilin_upsampler(x)\
        .transpose((0, 2, 3, 1))

cmap = plt.get_cmap('tab10')
data = {
    'truth': {
        'data': np.nan_to_num(y),
        'time': t,
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
        'time': t,
        'plotkwargs': {
            'label': 'SRResNet prediction',
            'linestyle': '-',
            'color': cmap(1),
            'zorder': 5,
        },
    },

    'pred_hybrid': {
        'data': np.nan_to_num(z_hybrid[0]),
        'time': t,
        'plotkwargs': {
            'label': 'ESNc prediction',
            'linestyle': '-',
            'color': cmap(2),
            'zorder': 4,
        },
    },
    'pred_hybrid2': {
        'data': np.nan_to_num(z_hybrid[1]),
        'time': t,
        'plotkwargs': {
            'label': 'ESNc prediction',
            'linestyle': '-',
            'color': cmap(6),
            'zorder': 4,
        },
    },
    'pred_hybrid3': {
        'data': np.nan_to_num(z_hybrid[2]),
        'time': t,
        'plotkwargs': {
            'label': 'ESNc prediction',
            'linestyle': '-',
            'color': cmap(8),
            'zorder': 4,
        },
    },
}

plt.switch_backend('qtagg')

for direction in ['temporal', 'spatial']:
    for spectrum_type in ['energy', 'enstrophy', 'ssh']:
        plot_machine.plot_spectrum(data,
                                   transect_name='along_flow',
                                   spectrum_type=spectrum_type,
                                   direction=direction,
                                   add_powerlaws=False)
                                   
plt.pause(1)
