import numpy as np
import compute_tool
import tools
import os
import dill
import itertools
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


class Metrics():

    def __init__(
            self,
            dm,
            ct,
            modes,
            metrics_file,
            reset=False,
    ):
        self.dm = dm
        self.ct = ct
        self.modes = modes
        self.metrics_file = metrics_file
        self.ct = compute_tool.ComputeTool(dm=self.dm)

        self.load_metrics_dict(reset=reset)

        self.trunc_time = 24*7

    def load_metrics_dict(self, reset=False):

        if not reset and os.path.exists(self.metrics_file):
            print('loading metrics dict')
            with open(self.metrics_file, 'rb') as file:
                self.metrics_dict = dill.load(file)
        else:
            self.metrics_dict = {}

        # initialize keys if needed
        if 'RMSE' not in self.metrics_dict:
            self.metrics_dict['RMSE'] = {}
        if 'correlation' not in self.metrics_dict:
            self.metrics_dict['correlation'] = {}
        if 'LSD' not in self.metrics_dict:
            self.metrics_dict['LSD'] = {}
        if 'DKL' not in self.metrics_dict:
            self.metrics_dict['DKL'] = {}
        if 'DKL_vals' not in self.metrics_dict:
            self.metrics_dict['DKL_vals'] = {}

    def save_metrics_dict(self):
        print('saving metrics dict')
        with open(self.metrics_file, 'wb') as file:
            dill.dump(self.metrics_dict, file)

    def field_manip(self, data, field_type='all'):
        if field_type == 'uo':
            return data[..., 0]
        elif field_type == 'ssh':
            return data[..., 2]
        elif field_type == 'all':
            return data
        elif field_type == 'vorticity':
            return self.ct.vorticity(data, None)
        elif field_type == 'enstrophy':
            return np.square(self.ct.vorticity(data, None))
        elif field_type == 'energy':
            return np.sum(np.square(data[..., :2]), axis=-1)
        else:
            raise Exception('invalid field_type')

    def compute_metric(
            self,
            data,
            metric='RMSE',
            field_type='all',
            **kwargs,
    ):
        if metric == 'LSD':  # log-spectrum distance
            self.compute_LSD(data, field_type, **kwargs)
        elif metric == 'DKL':  # KL distance
            self.compute_DKL(data, field_type, **kwargs)
        else:
            truth = self.field_manip(data['truth']['data'], field_type)
            self.modes_U = self.field_manip(self.modes['U'], field_type)
            for key, value in data.items():
                if key == 'truth':
                    continue

                prediction = self.field_manip(value['data'], field_type)

                if metric == 'RMSE':
                    self.compute_RMSE(truth, prediction, key, field_type)
                elif metric == 'correlation':
                    self.compute_correlation(truth,
                                             prediction,
                                             key,
                                             field_type)

    def compute_DKL(self, data, field_type, **kwargs):
        """ compute Kullback Leibler distance DKL """

        transect = kwargs['transect']
        T = {}  # transects
        for key, value in data.items():
            print(f'computing hovmöller {transect}, {field_type}, {key}')
            T[key] = self.ct.hovmöller_along_transect(
                value['data'],
                transect_name=transect,
                spectrum_type=field_type
            )
            if field_type == 'enstrophy':
                T[key] = np.square(T[key])

        operation = 'mean'
        ref_key = 'truth'
        bins = 500

        ref_vals = self.reduce(T[ref_key], operation)

        ref_std = np.std(ref_vals)
        # if field_type in ['MKE', 'TKE', 'enstrophy']:
        #     one_sided = True,
        # else:
        #     one_sided = False,

        xmin = np.min(ref_vals) - 2 * ref_std
        xmax = np.max(ref_vals) + 2 * ref_std
        # xmin = np.max([xmin, 0.0]) if one_sided else xmin
        ref_x = np.linspace(xmin, xmax, bins + 1)

        ref_pdf = self.compute_discrete_pdf(ref_vals, ref_x)

        # store reference values
        field_key = '_'.join([field_type, transect])
        if 'DKL_ref' in self.metrics_dict:
            self.metrics_dict['DKL_ref'].update(
                {'vals_' + field_key: ref_vals,
                 'x_' + field_key: ref_x})
        else:
            self.metrics_dict['DKL_ref'] = \
                {'vals_' + field_key: ref_vals,
                 'x_' + field_key: ref_x}

        dkl = {}
        pdf = {}
        pdf[ref_key] = ref_pdf

        # plt.close('all')

        # plt.figure()
        # plt.plot(ref_x[1:], ref_pdf / np.diff(ref_x)[0], 'k', label=ref_key)
        # plt.plot(ref_x, compute_kde(ref_vals, ref_x), 'r', label=ref_key)
        # plt.pause(1)

        # plt.figure()
        # plt.plot(ref_x, compute_kde(ref_vals, ref_x), 'k', label=ref_key)

        for key, value in T.items():
            if key == ref_key:
                continue
            vals = self.reduce(value, operation)
            pdf[key] = self.compute_discrete_pdf(vals, ref_x)
            dkl[key] = self.compute_dkl(pdf[ref_key], pdf[key])

            if key in self.metrics_dict['DKL']:
                self.metrics_dict['DKL'][key].update({field_key: dkl[key]})
            else:
                self.metrics_dict['DKL'][key] = {field_key: dkl[key]}

            if key in self.metrics_dict['DKL_vals']:
                self.metrics_dict['DKL_vals'][key].update({field_key: vals})
            else:
                self.metrics_dict['DKL_vals'][key] = {field_key: vals}

            # plt.plot(ref_x, compute_kde(vals, ref_x), label=key)
            # plt.plot(ref_x[1:], pdf[key] / np.diff(ref_x)[0], label=key)

        # plt.legend()
        # plt.pause(1)

    def reduce(self, mat, operation):
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

    def compute_discrete_pdf(self, vals, x):
        pdf, _ = np.histogram(vals, x, density=True)
        return pdf * np.diff(x)

    def compute_dkl(self, P, Q):
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

    def compute_LSD(self, data, field_type, **kwargs):
        """ compute log-spectrum distance LSD """

        # get true spectrum
        k = {}
        S = {}
        transect = kwargs['transect']
        direction = kwargs['direction']
        for key, value in data.items():
            print(f'computing hovmöller for {key}')
            k[key], S[key], _ = self.ct.compute_spectrum_along_transect(
                value['data'],
                transect_name=transect,
                spectrum_type=field_type,
                direction=direction,
            )

        # mean over space ortime
        S = {key: np.mean(value, axis=-1) for key, value in S.items()}

        field_type = '_'.join([field_type, transect, direction])
        for key, value in S.items():
            if key == 'truth':
                continue

            # discrete log-spectral distance LSD
            p = 2
            LSD = (np.sum((np.log(S[key]/S['truth']))**p) / len(S[key]))**(1/p)

            if key in self.metrics_dict['LSD']:
                self.metrics_dict['LSD'][key].update({field_type: LSD})
            else:
                self.metrics_dict['LSD'][key] = {field_type: LSD}

    def compute_RMSE(self, truth, prediction, key, field_type):
        shape = truth.shape
        # put spatial dim invector form
        error = (truth - prediction).reshape(shape[0], -1)
        # truncate first week to get rid of startup effects
        error = error[self.trunc_time:,]
        # sum over space
        errnorm = np.sum(np.square(error), -1)
        # mean over time
        RMSE = np.sqrt(np.mean(errnorm))
        # add to dict
        if key in self.metrics_dict['RMSE']:
            self.metrics_dict['RMSE'][key].update({field_type: RMSE})
        else:
            self.metrics_dict['RMSE'][key] = {field_type: RMSE}

    def compute_correlation(self, truth, prediction, key, field_type):
        shape = truth.shape

        correlations = []
        for mode in range(10):
            U = (self.modes_U.reshape(-1, np.prod(shape[1:])))[mode]
            # truncate, ignore first t steps
            pred = prediction[self.trunc_time:, ]\
                .reshape(-1, np.prod(shape[1:]))
            true = truth[self.trunc_time:, ]\
                .reshape(-1, np.prod(shape[1:]))

            pred_proj = U @ pred.T
            true_proj = U @ true.T

            correlations.append(
                np.corrcoef(np.vstack([pred_proj, true_proj]))[1, 0]
            )

        if key in self.metrics_dict['correlation']:
            self.metrics_dict['correlation'][key]\
                .update({field_type: correlations})
        else:
            self.metrics_dict['correlation'][key] = \
                {field_type: correlations}


def make_kdeplots(metrics_dict,
                  metric,
                  field_type,
                  base_dir,
                  **kwargs):

    mdict = metrics_dict[metric]
    keys = mdict.keys()
    ensembles, normal_runs =\
        tools.split_ensembles(keys)

    # create subset and change key name if needed
    if metric == 'DKL_vals':
        transect = kwargs['transect']
        field_key = '_'.join([field_type, transect])
        ref_vals = metrics_dict['DKL_ref']['vals_' + field_key]
        ref_x = metrics_dict['DKL_ref']['x_' + field_key]
    else:
        ref_vals = []
        ref_x = []
        raise Exception('Not implemented')

    subset = {}
    for key, value in ensembles.items():
        subset[key.replace('8', '08')] = \
            [mdict[mem][field_key] for mem in value]

    for run in normal_runs:
        subset[run.replace('8', '08')] = \
            [mdict[run][field_key]]

    kd_est_ref = compute_kde(ref_vals, ref_x)
    plt.fill_between(ref_x,
                     kd_est_ref,
                     kd_est_ref * 0.0,
                     color='k',
                     alpha=0.7,
                     zorder=0)

    plt.plot(ref_x, kd_est_ref,
             'k', label='reference',
             linewidth=2)
    labels = {'pred_resnet_cf32': 'SRResNet',
              'pred_esnc_cf32': 'CAE-ESNc',
              'bilin_cf32': 'bilinear interpolation'}
    cmap = plt.get_cmap('tab10')
    colors = {
        'pred_resnet_cf32': cmap(1),
        'pred_esnc_cf32': cmap(2),
        'bilin_cf32': cmap(0)}

    for key, value in subset.items():
        if '32' not in key:
            continue
        vals = value[0]
        kd_est = compute_kde(vals, ref_x)
        plt.fill_between(ref_x,
                         kd_est,
                         kd_est * 0.0,
                         color=colors[key],
                         alpha=0.4,
                         zorder=0)
        plt.plot(ref_x,
                 kd_est,
                 color=colors[key],
                 label=labels[key],
                 linewidth=2)

    plt.xlabel('PDF, ' + field_type)
    if field_type == 'ssh':
        plt.xlim(None, None)
    elif field_type == 'enstrophy':
        plt.xlim(0, 50)
    else:
        plt.xlim(0, None)

    plt.yticks([])
    plt.legend()
    plt.pause(1)


def compute_kde(vals, ref_x):
    xmin = np.min(ref_x)
    xmax = np.max(ref_x)
    kde = KernelDensity(
        kernel='linear',
        bandwidth=(xmax-xmin)/30,
    ).fit(vals[:, np.newaxis])
    log_densities = kde.score_samples(ref_x[:, np.newaxis])
    return np.exp(log_densities)


def make_boxplots(metrics_dict,
                  metric,
                  field_type,
                  base_dir,
                  plot_legend=True,
                  save_fig=True,
                  **kwargs):

    mdict = metrics_dict[metric]
    keys = mdict.keys()
    ensembles, normal_runs =\
        tools.split_ensembles(keys)

    # create subset and change key name if needed
    field_key = field_type
    if metric == 'LSD':
        transect = kwargs['transect']
        direction = kwargs['direction']
        field_key = '_'.join([field_type, transect, direction])
    elif metric == 'DKL':
        transect = kwargs['transect']
        field_key = '_'.join([field_type, transect])

    subset = {}
    for key, value in ensembles.items():
        subset[key.replace('8', '08')] = \
            [mdict[mem][field_key] for mem in value]

    for run in normal_runs:
        subset[run.replace('8', '08')] = \
            [mdict[run][field_key]]

    sorted_keys = sorted(subset.keys())

    lstyles = ['-', ':', '--']
    lstyles_cycler = itertools.cycle(lstyles)

    cfs = [key[-2:] for key in sorted_keys]
    grouped = {}
    for (cf, key) in zip(cfs, sorted_keys):
        grouped[cf] = grouped[cf]+[key] if cf in grouped else [key]

    Nmodels = len(list(grouped.values())[0])
    Ncfs = len(list(grouped.keys()))

    cmap = plt.get_cmap('tab10')
    if metric == 'correlation':
        colors = [*[cmap(0)]*Ncfs, *[cmap(2)]*Ncfs, *[cmap(1)]*Ncfs]
    elif metric in ['RMSE', 'LSD', 'DKL']:
        colors = [cmap(0), cmap(2), cmap(1)]

    labels = {}
    for key in sorted_keys:
        key_orig = key
        if 'bilin' in key:
            key = key.replace('bilin_', 'bilinear interpolation,')
        if 'esnc' in key:
            key = key.replace('pred_esnc_', 'CAE-ESNc,')
        if 'resnet' in key:
            key = key.replace('pred_resnet_', 'SRResNet,')
        if '08' in key:
            key = key.replace('cf08', ' $CF=8$')
        if '16' in key:
            key = key.replace('cf16', ' $CF=16$')
        if '32' in key:
            key = key.replace('cf32', ' $CF=32$')
        labels[key_orig] = key

    if metric == 'correlation':
        plt.figure()
        data = [np.asarray(subset[key]) for key in sorted_keys]
        for (value, key, col) in zip(data, sorted_keys, colors):
            lstyle = next(lstyles_cycler)
            q1 = np.quantile(value, 0.25, axis=0)
            q2 = np.quantile(value, 0.5, axis=0)
            q3 = np.quantile(value, 0.75, axis=0)
            if not np.all(q1 == q3):
                plt.fill_between(range(10), q1, q3, color=col,
                                 zorder=0, alpha=0.5, linestyle=lstyle)

            plt.plot(range(10), q2, label=labels[key],
                     color=col, linewidth=2.5,
                     linestyle=lstyle)

        plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.02),
                   loc='lower center', borderaxespad=0)
        plt.ylabel('correlation')
        plt.xlabel('PC')
        plt.grid(which='both')
        fig_name = f'{base_dir}/correlations_{field_key}.png'
        print(fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=200)

    elif metric in ['RMSE', 'LSD', 'DKL']:
        q1 = [np.quantile(subset[key], 0.25) for key in sorted_keys]
        q2 = [np.quantile(subset[key], 0.50) for key in sorted_keys]
        q3 = [np.quantile(subset[key], 0.75) for key in sorted_keys]
        all_vals = [subset[key] for key in sorted_keys]

        if metric == 'RMSE':
            plt.figure(figsize=(3.5, 4.8))
        elif metric == 'LSD' and save_fig:
            plt.figure(figsize=(2.5, 4.8))

        for i, col, label in zip(range(Nmodels), colors,
                                 ['bilin. interp.',
                                  'CAE-ESNc',
                                  'SRResNet']):
            r = slice(i*Ncfs, i*Ncfs+Ncfs)

            plt.fill_between(range(Ncfs), q1[r], q3[r],
                             color=col,
                             zorder=0, alpha=0.5,
                             )
            pos = range(Ncfs)
            plt.plot(pos, q2[r], '.-',
                     color=col,
                     linewidth=2.5,
                     markersize=10,
                     label=label,
                     )
            if not np.all(q1[r] == q3[r]):
                plt.boxplot(all_vals[r],
                            positions=pos,
                            showfliers=False,
                            )
                ylabel = {
                    'all': f'{metric}, total',
                    'uo': f'{metric}, zonal velocity',
                    'ssh': f'{metric}, SSH'
                }
        if field_type in ylabel:
            plt.ylabel(ylabel[field_type])

        labels = ['$CF=8$', '$CF=16$', '$CF=32$']
        plt.gca().set_xticks([0, 1, 2])
        rotation = 0 if metric == 'RMSE' else 45
        plt.gca().set_xticklabels(labels, rotation=rotation)
        if metric in ['LSD', 'DKL']:
            plt.yscale('log')

        plt.grid(which='both')

        if plot_legend:
            plt.legend()

        if save_fig:
            fig_name = f'{base_dir}/{metric}_{field_key}.png'
            print(fig_name)
            plt.savefig(fig_name, bbox_inches='tight', dpi=200)
