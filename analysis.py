# compare runs
import plot_utils
import importlib
import dill
import numpy as np
import tools
import data_manager_cmems
from sklearn.neighbors import KernelDensity
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


def load_timeseries(fname):  # TODO this needs a better implementation
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
    # 'experiment/predictor_ESNcT5e-2_6mpred_ks6000/results',
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

    'lowres': {
        'data': np.nan_to_num(x),
        'time': [],
        'plotkwargs': {
            'label': 'bilinear interpolation',
            'linestyle': '--',
            'color': cmap(5),
            'zorder': 0,
        },
    },

    'pred_resnet': {
        'data': np.nan_to_num(z_resnet),
        'time': t,
        'plotkwargs': {
            'label': 'SRResNet',
            'linestyle': '-',
            'color': cmap(1),
            'zorder': 5,
        },
    },

    'pred_hybrid': {
        'data': np.nan_to_num(z_hybrid[0]),
        'time': t,
        'plotkwargs': {
            'label': 'CAE+ESNc, $\lambda=0.005$',
            'linestyle': '-',
            'color': cmap(2),
            'zorder': 4,
        },
    },
    'pred_hybrid2': {
        'data': np.nan_to_num(z_hybrid[1]),
        'time': t,
        'plotkwargs': {
            'label': 'CAE+ESNc, $\lambda=0.01$',
            'linestyle': '-',
            'color': cmap(6),
            'zorder': 4,
        },
    },
    # 'pred_hybrid3': {
    #     'data': np.nan_to_num(z_hybrid[2]),
    #     'time': t,
    #     'plotkwargs': {
    #         'label': 'ESNc prediction',
    #         'linestyle': '-',
    #         'color': cmap(8),
    #         'zorder': 4,
    #     },
    # },
}

plt.switch_backend('qtagg')

for spectrum_type in ['energy', 'enstrophy', 'ssh']:
    for direction in ['temporal']:
        S, T = plot_machine.plot_spectrum(data,
                                          transect_name='along_flow',
                                          spectrum_type=spectrum_type,
                                          direction=direction,
                                          add_powerlaws=False)
        plt.pause(.1)

breakpoint()

TKE = {}
MKE = {}
KE = {}
SSH = {}
Zens = {}
for key, value in data.items():
    TKE[key] = \
        plot_machine.ct.hovmöller_along_transect(value['data'],
                                                 spectrum_type='TKE')
    SSH[key] = \
        plot_machine.ct.hovmöller_along_transect(value['data'],
                                                 spectrum_type='ssh')
    KE[key] = \
        plot_machine.ct.hovmöller_along_transect(value['data'],
                                                 spectrum_type='energy')
    MKE[key] = \
        plot_machine.ct.hovmöller_along_transect(value['data'],
                                                 spectrum_type='MKE')
    Zens[key] = \
        plot_machine.ct.hovmöller_along_transect(value['data'],
                                                 spectrum_type='enstrophy')**2


def hist_plot(vec, color, label):
    n, bins, _ = plt.hist(vec, bins=100, density=True, color=color, alpha=0.5)
    plt.plot(bins[1:] - (bins[1]-bins[0])/2,
             n,
             color=color,
             linewidth=2,
             label=label)


def reduce(mat, operation):

    if operation == 'sum':
        vec = np.sum(mat, -1)
    elif operation == 'mean':
        vec = np.mean(mat, -1)
    elif operation == 'first':
        vec = mat[:, 0]
    elif operation == 'middle':
        vec = mat[:, int(mat.shape[1] / 2)]
    elif operation == 'last':
        vec = mat[:, -1]
    else:
        raise Exception('invalid operation')

    return vec


def plot_histograms(input_dict, hist_type='hist', operation='sum'):
    plt.figure()
    cmap = plt.get_cmap('tab10')

    minval = 0
    maxval = 0

    for key, value in input_dict.items():
        vec = np.sum(value, -1)
        minval = np.min([np.min(vec), minval])
        maxval = np.max([np.max(vec), maxval])

    for idx, (key, value) in enumerate(input_dict.items()):
        # vec = (np.sum(value, -1) - minval) / (maxval - minval)

        color = cmap(idx)

        if hist_type == 'kde':
            datavec = vec[:, np.newaxis]

            kde = \
                KernelDensity(kernel='gaussian',
                              bandwidth=(maxval-minval)/100).fit(datavec)

            x_plot = np.linspace(np.min(datavec),
                                 np.max(datavec),
                                 1000)[:, np.newaxis]

            y = kde.score_samples(x_plot)
            plt.plot(x_plot, np.exp(y), label=key, color=color)
        elif hist_type == 'hist':
            hist_plot(vec, color, key)

    plt.legend()


input_dict = MKE
operation = 'sum'
ref_key = 'truth'
bins = 1000

ref_vals = reduce(input_dict[ref_key], operation)

# setup interval, 2*sigma outside of reference domain
ref_mn = np.mean(ref_vals)
ref_std = np.std(ref_vals)
ref_x = np.linspace(np.min(ref_vals) - 2 * ref_std,
                    np.max(ref_vals) + 2 * ref_std,
                    bins + 1)


def compute_discrete_pdf(vals, x):
    pdf, _ = np.histogram(vals, x, density=True)
    return pdf * np.diff(x)


ref_pdf = compute_discrete_pdf(ref_vals, ref_x)


def compute_dkl(P, Q):
    eps = 1e-16

    # add eps
    P += eps
    Q += eps

    # do some checks
    assert P.shape == Q.shape, "incompatible shapes"
    assert (np.sum(P) - 1.0) < 1e-11, "input not a pdf"
    assert (np.sum(Q) - 1.0) < 1e-11, "input not a pdf"

    out = np.sum(P * np.log(P / Q))
    return out


for key, value in input_dict.items():

    vals = reduce(value, operation)
    pdf = compute_discrete_pdf(vals, ref_x)
    dkl = compute_dkl(ref_pdf, pdf)
    print(key, dkl)
