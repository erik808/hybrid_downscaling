import plot_utils
import importlib
import os
import itertools
import time
import keras
import dill
import numpy as np
import tools
import stats_tool
import data_manager_cmems
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
importlib.reload(data_manager_cmems)
importlib.reload(plot_utils)
importlib.reload(stats_tool)
importlib.reload(tools)
plt.switch_backend('qtagg')


class Analysis():

    def __init__(self):
        self.reference_path = \
            'experiment/reference/truths.dill'

        self.resnet_path = lambda cf, mem: \
            ('experiment/'
             f'resnet_subpixel_prelu_b6f64o0_cf{cf}/member_{mem}/'
             'results/predictions.dill')

        self.esnc_path = lambda cf, mem: \
            ('experiment/predESNc_rho1_lam1e-2_hist5000_'
             f'vaef64-64_cf{cf}/member_{mem}/'
             'results/predictions.dill')

        self.dmgr_cmems = \
            data_manager_cmems.DataManagerCMEMS(
                experiment_id='analysis',
                testing=False,
                force_rebuild=False,
                base_dir=".",
            )

        self.plot_machine = plot_utils.PlotMachine(dm=self.dmgr_cmems)

        results_dir_org = self.plot_machine.results_dir
        self.results_dir = results_dir_org + '/merge'
        self.plot_machine.set_results_dir(self.results_dir)

        self.HR_scaler = self.dmgr_cmems.scalers['HR']

        self.y_truth = self.load_reference()
        modes = self.get_modes()
        self.metrics = stats_tool.Metrics(
            dm=self.dmgr_cmems, modes=modes)

    def get_modes(self, force_compute=False):
        self.modes_file = f'{self.results_dir}/modes.dill'
        if os.path.exists(self.modes_file) and not force_compute:
            print('modes file exists, loading')
            with open(self.modes_file, 'rb') as file:
                modes = dill.load(file)
        else:
            print('computing modes...')
            modes = self.compute_modes()
        return modes

    def compute_modes(self):

        print('computing SVD... ', end="")
        tdim = self.y_truth.shape[0]

        tic = time.time()

        X = np.nan_to_num(self.y_truth.reshape(tdim, -1).T)
        # mean center
        X = (X - np.mean(X, axis=0)) / np.sqrt(tdim - 1)

        U, s, Vt = np.linalg.svd(
            X,
            full_matrices=False,
            compute_uv=True)

        print('done', end="")
        toc = time.time()
        print(f' {toc-tic}')

        modes = {'U': (U.T).reshape(self.y_truth.shape),
                 's': s,
                 'Vt': Vt}

        with open(self.modes_file, 'wb') as file:
            dill.dump(modes, file)

        return modes

    def load_reference(self):
        print('loading reference')
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

        LR_input_orig = tools.unscale_var(LR_input, LR_scaler)
        LR_input = np.ascontiguousarray(LR_input_orig.transpose((0, 3, 1, 2)))
        LR_input = dmgr_cmems_tmp\
            .bilin_upsampler(LR_input)\
            .transpose((0, 2, 3, 1))
        return LR_input, {'LR_input_orig': LR_input_orig,
                          'LR_grid': dmgr_cmems_tmp.grid_LR}

    def load_time(self):
        with open(self.resnet_path(32, 0), 'rb') as file:
            results = dill.load(file)['results']
        time = np.array([np.datetime64(re['time']) for re in results])
        return time

    def load_bilin(self, cf_vals):
        out_bilin = {}
        out_orig = {}
        print('loading bilinear interpolation results')
        pb_i = keras.utils.Progbar(len(cf_vals), interval=0.5)
        for cf in cf_vals:
            pb_i.add(1)
            out_bilin[cf], out_orig[cf] = \
                self.load_input(self.resnet_path(cf, 0))
        return out_bilin, out_orig

    def load_resnet(self, cf_vals, members):
        print('loading resnet results')
        pb_i = keras.utils.Progbar(
            len(cf_vals) * len(members), interval=0.5)

        out = {}
        for cf in cf_vals:
            out[cf] = {}
            for member in members:
                pb_i.add(1)
                out[cf][member] = analyzer.load_prediction(
                    self.resnet_path(cf, member))
        return out

    def load_esnc(self, cf_vals, members):
        print('loading esnc results')
        pb_i = keras.utils.Progbar(
            len(cf_vals) * len(members), interval=0.5)

        out = {}
        for cf in cf_vals:
            out[cf] = {}
            for member in members:
                fname = self.esnc_path(cf, member)
                pb_i.add(1)
                out[cf][member] = \
                    analyzer.load_prediction(fname)

        return out

    def create_data_dict(
            self,
            cf_vals,
            members,
            z_bilin,
            z_input,
            z_resnet,
            z_esnc,
    ):
        time = self.load_time()
        cmap = plt.get_cmap('tab10')
        data = {
            'truth': {
                'data': np.nan_to_num(self.y_truth),
                'time': time,
                'plotkwargs': {
                    'label': 'reference',
                    'linestyle': '-',
                    'linewidth': 3,
                    'color': 'k',
                    'zorder': 0,
                },
            },
        }

        for cf in cf_vals:
            data.update(
                {
                    f'bilin_cf{cf}': {
                        'data': np.nan_to_num(z_bilin[cf]),
                        'input': np.nan_to_num(z_input[cf]),
                        'time': time,
                        'cf': cf,
                        'plotkwargs': {
                            'label': 'bilinear interpolation',
                            'linestyle': ':',
                            'zorder': 2,
                            'linewidth': 2,
                            'color': cmap(0),
                        },
                    },
                }
            )

            for member in members:

                data.update(
                    {
                        f'pred_resnet_cf{cf}/m{member}': {
                            'data': np.nan_to_num(z_resnet[cf][member]),
                            'time': time,
                            'cf': cf,
                            'plotkwargs': {
                                'label': 'SRResNet',
                                'linestyle': '-',
                                'zorder': 7,
                                'linewidth': 2,
                                'color': cmap(1),
                            },
                        },
                    }
                )
                data.update(
                    {
                        f'pred_esnc_cf{cf}/m{member}': {
                            'data': np.nan_to_num(z_esnc[cf][member]),
                            'time': time,
                            'cf': cf,
                            'plotkwargs': {
                                'label': 'CAE-ESNc',
                                'linestyle': '--',
                                'zorder': 8,
                                'linewidth': 2,
                                'color': cmap(2),
                            },
                        },
                    }
                )
        return data


analyzer = Analysis()
members = range(10)
cfrange = [8, 16, 32]

# members = [0]
# cfrange = [32]

plot_legend = True

for cf in cfrange:
    plt.close('all')
    z_bilin, z_input = analyzer.load_bilin([cf])
    z_resnet = analyzer.load_resnet([cf], members)
    z_esnc = analyzer.load_esnc([cf], members)
    data_dict = analyzer.create_data_dict(
        [cf],
        members,
        z_bilin,
        z_input,
        z_resnet,
        z_esnc)

    # RMSE and other statistics
    ftypes = ['uo', 'ssh', 'all', 'vorticity', 'energy']
    metrics = ['RMSE', 'correlation']
    for metric in metrics:
        for field_type in ftypes:
            analyzer.metrics.compute_metric(
                data_dict,
                metric=metric,
                field_type=field_type)
    continue

    # only case to do reconstruction plotting
    if members == [0] and cf == 32:
        analyzer.plot_machine.plot_hovmöller(data_dict,
                                             compute=True,
                                             plot_type='energy',
                                             transect='along_flow')

        # 2d coarse input plots
        analyzer.plot_machine.plot_coarse_input(
            data_dict, field_type='uo')

        # 2d overview plots
        analyzer.plot_machine.plot_2d_fields(
            data_dict, field_type='uo', overview=True)
        analyzer.plot_machine.plot_2d_fields(
            data_dict, field_type='energy', overview=True)
        analyzer.plot_machine.plot_2d_fields(
            data_dict, field_type='vorticity', overview=True)

        # plot 2d fields
        analyzer.plot_machine.plot_2d_fields(
            data_dict, field_type='energy', overview=False)
        analyzer.plot_machine.plot_2d_fields(
            data_dict, field_type='vorticity', overview=False)
        analyzer.plot_machine.plot_2d_fields(
            data_dict, field_type='uo', overview=False)

    # plot spectra
    transect = 'along_flow'
    for direction in ['temporal', 'spatial']:
        for spectrum_type in ['energy', 'enstrophy', 'ssh']:
            plt.figure(figsize=(5, 3.5))
            S, T = analyzer.plot_machine.plot_spectrum(
                data_dict,
                transect_name=transect,
                spectrum_type=spectrum_type,
                direction=direction,
                add_powerlaws=False,
                make_title=False,
                plot_legend=plot_legend)
            plt.close('all')

    plot_legend = False
    # cleanup
    del z_bilin, z_resnet, z_esnc, data_dict

# manipulate metrics dict
metric = 'correlation'
field_type = 'all'

mdict = analyzer.metrics.metrics_dict[metric]
keys = mdict.keys()
ensembles, normal_runs = \
    tools.split_ensembles(keys)
all_runs = normal_runs + [key
                          for key in ensembles.keys()]

d = {}

for key, value in ensembles.items():
    d[key.replace('8', '08')] = [mdict[mem][field_type] for mem in value]

for run in normal_runs:
    d[run.replace('8', '08')] = [mdict[run][field_type]]

sorted_keys = sorted(d.keys())

lstyles = ['-', ':', '-.']
lstyles_cycler = itertools.cycle(lstyles)

cfs = [key[-2:] for key in sorted_keys]
grouped = {}
for (cf, key) in zip(cfs, sorted_keys):
    grouped[cf] = grouped[cf]+[key] if cf in grouped else [key]

Nmodels = len(list(grouped.values())[0])
Ncfs = len(list(grouped.keys()))
cmap = plt.get_cmap('tab10')
colors = [*[cmap(0)]*Ncfs, *[cmap(2)]*Ncfs, *[cmap(1)]*Ncfs]

labels = {}
for key in sorted_keys:
    key_orig = key
    if 'bilin' in key:
        key = key.replace('bilin_', 'bilin')
    if 'esnc' in key:
        key = key.replace('pred_esnc_', 'CAE-ESNc')
    if 'resnet' in key:
        key = key.replace('pred_resnet_', 'SRResNet')
    if '08' in key:
        key = key.replace('cf08', ' $CF=8$')
    if '16' in key:
        key = key.replace('cf16', ' $CF=16$')
    if '32' in key:
        key = key.replace('cf32', ' $CF=32$')
    labels[key_orig] = key

if metric == 'correlation':
    plt.figure(figsize=(8, 6))
    data = [np.asarray(d[key]) for key in sorted_keys]
    for (value, key, col) in zip(data, sorted_keys, colors):
        lstyle = next(lstyles_cycler)
        q1 = np.quantile(value, 0.25, axis=0)
        q2 = np.quantile(value, 0.5, axis=0)
        q3 = np.quantile(value, 0.75, axis=0)
        if not np.all(q1 == q3):
            plt.fill_between(range(10), q1, q3, color=col,
                             zorder=0, alpha=0.5, linestyle=lstyle)

        plt.plot(range(10), q2, label=labels[key],
                 color=col, linewidth=2,
                 ncol=2,
                 linestyle=lstyle)

    plt.legend()
    plt.ylabel('correlation')
    plt.xlabel('PC')
    plt.grid(which='both')
    fig_name = f'{analyzer.results_dir}/correlations.png'
    print(fig_name)
    plt.savefig(fig_name, bbox_inches='tight', dpi=200)
    plt.pause(1)

elif metric == 'RMSE':
    plt.figure()
    plt.boxplot([d[key] for key in sorted_keys],
                tick_labels=sorted_keys,
                vert=False)
    plt.gca().set_title(f'{metric}, {field_type}')
    plt.pause(1)

    q0 = [np.quantile(d[key], 0.0) for key in sorted_keys]
    q1 = [np.quantile(d[key], 0.25) for key in sorted_keys]
    q2 = [np.quantile(d[key], 0.50) for key in sorted_keys]
    q3 = [np.quantile(d[key], 0.75) for key in sorted_keys]
    q4 = [np.quantile(d[key], 1.0) for key in sorted_keys]

    all_vals = [d[key] for key in sorted_keys]

    plt.figure()
    for i, col, label in zip(range(Nmodels), colors,
                             ['bilinear interpolation',
                              'CAE-ESNc',
                              'SRResNet']):
        r = slice(i*Ncfs, i*Ncfs+Ncfs)
        print(r)
        plt.fill_between(range(Ncfs), q1[r], q3[r],
                         color=col,
                         zorder=0, alpha=0.5,
                         label=label)
        pos = range(Ncfs)
        plt.plot(pos, q2[r], '.-',
                 color=col,
                 linewidth=2
                 )
        plt.boxplot(all_vals[r], positions=pos)
    plt.ylabel('RMSE')
    plt.xlabel('CF')
    plt.xticks([8, 16, 32])
    plt.legend()
    plt.pause(1)









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
