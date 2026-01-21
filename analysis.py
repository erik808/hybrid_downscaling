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


class Analysis():

    def __init__(self):
        self.reference_path = \
            'experiment/reference/truths.dill'

        self.dmgr_cmems = \
            data_manager_cmems.DataManagerCMEMS(
                experiment_id='analysis',
                testing=False,
                force_rebuild=False,
                base_dir=".",
            )
        self.plot_machine = plot_utils.PlotMachine(dm=self.dmgr_cmems)
        results_dir_org = self.plot_machine.results_dir
        results_dir = results_dir_org + '/merge'
        self.plot_machine.set_results_dir(results_dir)

        self.HR_scaler = self.dmgr_cmems.scalers['HR']

    def load_reference(self):
        with open(self.reference_path, 'rb') as file:
            truths = dill.load(file)['truths']

        HR_truth = \
            np.concatenate([tr['HR_data'][:, 0,] for tr in truths], 0)

        HR_truth = tools.unscale_var(HR_truth, self.HR_scaler)
        return HR_truth

    def load_prediction(self, fname):
        with open(fname, 'rb') as file:
            results = dill.load(file)['results']

        HR_pred = np.concatenate([re['HR_data'][:, 0,] for re in results], 0)
        HR_pred = tools.unscale_var(HR_pred, self.HR_scaler)
        return HR_pred

    def load_input(self, fname):

        with open(fname, 'rb') as file:
            results = dill.load(file)['results']

        LR_input = \
            np.concatenate([re['LR_data'][:, 0,] for re in results], 0)

        HR_pred = results[0]['HR_data'][:, 0,]
        cf = HR_pred.shape[1] / LR_input.shape[1]
        assert cf == HR_pred.shape[2] / LR_input.shape[2], \
            'nonsquare coarsening, not implemented'

        # create temporary datamanager to get scalers and upsamling
        dmgr_cmems_tmp = \
            data_manager_cmems.DataManagerCMEMS(
                force_coarsening_factor=int(cf)
            )
        LR_scaler = dmgr_cmems_tmp.scalers['LR']

        LR_input = tools.unscale_var(LR_input, LR_scaler)
        LR_input = np.ascontiguousarray(LR_input.transpose((0, 3, 1, 2)))
        LR_input = dmgr_cmems_tmp\
            .bilin_upsampler(LR_input)\
            .transpose((0, 2, 3, 1))
        return LR_input

    def load_time(self, fname):
        with open(fname, 'rb') as file:
            results = dill.load(file)['results']
        time = np.array([np.datetime64(re['time']) for re in results])
        return time


analyzer = Analysis()

y = analyzer.load_reference()

members = [4]
cf_vals = [4, 8, 16, 32]

z_resnet = {}
for cf in cf_vals:
    z_resnet[cf] = {}
    for member in members:
        z_resnet[cf][member] = analyzer.load_prediction(
            ('experiment/'
             f'resnet_bilinear_b6f64o0_cf{cf}/member_{member}/'
             'results/predictions.dill'),
        )

cmap = plt.get_cmap('tab10')
data = {
    'truth': {
        'data': np.nan_to_num(y),
        'time': [],
        'plotkwargs': {
            'label': 'high-resolution truth',
            'linestyle': '-',
            'linewidth': 2,
            'color': cmap(0),
            'zorder': 10,
        },
    },
}

col_ctr = 0
for cf in cf_vals:
    for member in members:
        col_ctr += 1
        data.update(
            {
                f'pred_resnet_cf{cf}m{member}': {
                    'data': np.nan_to_num(z_resnet[cf][member]),
                    'time': [],
                    'plotkwargs': {
                        'label': f'SRResNet cf{cf}m{member}',
                        'linestyle': '-',
                        'color': cmap(col_ctr),
                        'zorder': 5,
                    },
                },
            })

plt.switch_backend('qtagg')

# breakpoint()
plt.close('all')
for spectrum_type in ['energy', 'enstrophy', 'ssh']:  # , 'enstrophy', 'ssh']:
    for direction in ['temporal', 'spatial']:
        S, T = analyzer.plot_machine.plot_spectrum(data,
                                                   transect_name='along_flow',
                                                   spectrum_type=spectrum_type,
                                                   direction=direction,
                                                   add_powerlaws=False)
        plt.pause(.1)

raise Exception('que')

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
