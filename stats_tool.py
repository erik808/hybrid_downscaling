import numpy as np
import compute_tool
import tools
import itertools
import matplotlib.pyplot as plt


class Metrics():

    def __init__(self, dm, modes):
        self.dm = dm
        self.modes = modes
        self.ct = compute_tool.ComputeTool(dm=self.dm)
        self.metrics_dict = {}
        self.metrics_dict['RMSE'] = {}
        self.metrics_dict['correlation'] = {}
        self.trunc_time = 24*7

    def field_manip(self, data, field_type='all'):
        if field_type == 'uo':
            return data[..., 0]
        elif field_type == 'ssh':
            return data[..., 2]
        elif field_type == 'all':
            return data
        elif field_type == 'vorticity':
            return self.ct.vorticity(data, None)
        elif field_type == 'energy':
            return np.sum(np.square(data[..., :2]), axis=-1)
        else:
            raise Exception('invalid field_type')

    def compute_metric(self, data, metric='RMSE', field_type='all'):
        truth = self.field_manip(data['truth']['data'], field_type)
        self.modes_U = self.field_manip(self.modes['U'], field_type)

        for key, value in data.items():
            if key == 'truth':
                continue

            prediction = self.field_manip(value['data'], field_type)

            if metric == 'RMSE':
                self.compute_RMSE(truth, prediction, key, field_type)
            elif metric == 'correlation':
                self.compute_correlation(truth, prediction, key, field_type)

    def compute_RMSE(self, truth, prediction, key, field_type):
        shape = truth.shape
        # put spatial dim in vector form
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


def make_plots(metrics_dict, metric, field_type, base_dir):

    mdict = metrics_dict[metric]
    keys = mdict.keys()
    ensembles, normal_runs = \
        tools.split_ensembles(keys)

    # create subset and change key name if needed
    subset = {}
    for key, value in ensembles.items():
        subset[key.replace('8', '08')] = \
            [mdict[mem][field_type] for mem in value]

    for run in normal_runs:
        subset[run.replace('8', '08')] = \
            [mdict[run][field_type]]

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
    elif metric == 'RMSE':
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
        fig_name = f'{base_dir}/correlations_{field_type}.png'
        print(fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=200)
        
    elif metric == 'RMSE':

        q1 = [np.quantile(subset[key], 0.25) for key in sorted_keys]
        q2 = [np.quantile(subset[key], 0.50) for key in sorted_keys]
        q3 = [np.quantile(subset[key], 0.75) for key in sorted_keys]
        all_vals = [subset[key] for key in sorted_keys]

        plt.figure()
        for i, col, label in zip(range(Nmodels), colors,
                                 ['bilinear interpolation',
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

        ylabel = {'all': 'RMSE, total',
                  'uo': 'RMSE, zonal velocity',
                  'ssh': 'RMSE, SSH',
                  }
        plt.ylabel(ylabel[field_type])
        labels = ['$CF=8$', '$CF=16$', '$CF=32$']
        plt.gca().set_xticks([0, 1, 2])
        plt.gca().set_xticklabels(labels)
        plt.legend()
        plt.grid(which='both')
        fig_name = f'{base_dir}/RMSE_{field_type}.png'
        print(fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=200)
