from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import tools
from multiprocess import Pool
import importlib
from skimage.draw import line
from scipy.stats import binned_statistic
import data_utils
import compute_tool

importlib.reload(data_utils)
importlib.reload(compute_tool)


class PlotMachine():
    def __init__(self,
                 dirs,
                 output_dict={},
                 time_array=None,
                 trial_id=None,
                 figsize=(16, 8),
                 ):

        self.dirs = dirs
        self.figsize = figsize
        self.output_dict = output_dict
        self.time_array = time_array
        self.cbar_shrinkf = 0.5
        self.frame_stride = 4
        self.pool_size = 1
        self.trial_id = trial_id
        self.ct = compute_tool.ComputeTool()

    def plot_reconstructions(self, plot_dict):
        metadata = plot_dict['meta']
        plot_dict.pop('meta', None)
        postfix = \
            self.create_postfix(add_name=f"epoch{metadata['epoch']}")
        fig_name = f"{self.dirs['results']}/reconstructions{postfix}.png"

        num_plots = len(plot_dict.keys())
        M, N = 2, 3
        while M * N < num_plots:
            M += 1

        plt.figure(figsize=(N * 5, M * 3))
        for i, (key, item) in enumerate(plot_dict.items()):
            plt.subplot(M, N, i + 1)
            a = plt.pcolormesh(item['data'],
                               vmin=item['vmin'],
                               vmax=item['vmax'],
                               )
            plt.colorbar(a)
            plt.gca().set_title(key)

        print(f'saving to {fig_name}')
        plt.savefig(fig_name, bbox_inches='tight')

    def plot_single_frame(self, frame_id, output_dict=None):
        self.output_dict = self.output_dict \
            if output_dict is None else output_dict

        plt.figure(figsize=self.figsize)
        postfix = self.create_postfix()
        fig_name = f'{self.results_dir}/results_autoencoder{postfix}.png'
        print(fig_name)
        self.plot_frame(frame_id, fig_name)

    def create_movie(self, output_dict=None):

        self.output_dict = self.output_dict \
            if output_dict is None else output_dict

        plt.figure(figsize=self.figsize)

        if self.pool_size == 1:
            for i in range(0, len(self.time_array),
                           self.frame_stride):
                self.plot_frame(i)
        else:
            with Pool(self.pool_size) as p:
                p.map(self.plot_frame, range(0, len(self.time_array),
                                             self.frame_stride))

        postfix = self.create_postfix()
        movie_name = f'movie{postfix}.mov'
        framerate = 24
        sys_cmd = (f"ffmpeg -r {framerate} -f image2 -pattern_type glob -i "
                   f"'{self.movie_dir}/frame-*.png' "
                   f"-vcodec libx264 -crf 25  -pix_fmt yuv420p -y "
                   f"{self.movie_dir}/{movie_name}"
                   )

        print(sys_cmd)
        os.system(sys_cmd)
        sys_cmd = (f"rm {self.movie_dir}/frame-*.png")
        print(sys_cmd)
        os.system(sys_cmd)

    def plot_frame(self, id, fig_name=None):
        print(f'plotting frame {id}')
        plt.clf()
        if fig_name is None:
            fig_name = f'{self.movie_dir}/frame-{id:06d}.png'

        Nsub = len(self.output_dict)
        dim0 = int(np.ceil(np.sqrt(Nsub)))
        dim1 = int(np.ceil(Nsub / dim0))
        for f, (key, item) in enumerate(self.output_dict.items()):
            plt.subplot(dim1, dim0, f + 1)
            if item['type'] == '2d':
                h = plt.imshow(item['values'](id),
                               cmap=item['cmap'],
                               vmin=item['vmin'],
                               vmax=item['vmax'])
                plt.colorbar(h, shrink=self.cbar_shrinkf,
                             label=item['cbar_label'])
                plt.gca().set_title(key)
                plt.gca().invert_yaxis()

            elif (key == 'spectrum along flow' or
                  key == 'spectrum across flow'):
                for (name, var) in item['values'].items():
                    plt.loglog(var(id), '.-', label=name)

                plt.legend()
                plt.grid()
                plt.gca().set_title(key)
                plt.gca().set_aspect(0.2)
                plt.gca().set_ylim(item['ymin'], item['ymax'])
                plt.gca().set_xlim(item['xmin'], item['xmax'])

        plt.suptitle(f"date: {np.datetime64(self.time_array[id], 'h')}")
        plt.savefig(fig_name, bbox_inches='tight')

    def plot_history(self, hist, managed=False, add='', plot_baseline=True):

        if not managed:
            postfix = self.create_postfix()
            fig_name = f'{self.results_dir}/history{postfix}.png'
            plt.close('all')

        plt.subplot(1, 2, 1)
        for key, value in hist.history.items():
            if key == 'error' or key == 'base':
                continue
            plt.semilogy(value, '.-',
                         label=f'{key} {add}')

        plt.grid()
        plt.legend()
        plt.gca().set_xlabel('epoch')

        plt.subplot(1, 2, 2)
        if 'error' in hist.history:
            plt.semilogy(hist.history['error'], '.-',
                         label=f'validation error {add}')

        if 'base' in hist.history and plot_baseline:
            plt.semilogy(hist.history['base'], '.-',
                         label='validation baseline')

        plt.grid()
        plt.legend()
        plt.gca().set_xlabel('epoch')

        if not managed:
            print(fig_name)
            plt.tight_layout()
            plt.savefig(fig_name)

    def plot_prediction_error(self, X, Y, Z, add_name=''):

        postfix = self.create_postfix(add_name)
        fig_name = f'{self.results_dir}/errors{postfix}.png'

        RSE_Y = np.sqrt(np.sum(np.square(X - Y), axis=(1, 2, 3)))
        RSE_Z = np.sqrt(np.sum(np.square(X - Z), axis=(1, 2, 3)))

        plt.close('all')
        plt.plot(RSE_Y, label='RSE_Y')
        plt.plot(RSE_Z, label='RSE_Z')
        plt.grid()
        plt.legend()
        plt.gca().set_xlabel('time step')
        print(fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)

        return RSE_Y, RSE_Z

    def create_postfix(self, add_name=''):

        postfix = ''
        if self.trial_id is not None:
            postfix += f'_trial_{self.trial_id}'

        postfix += f'_{add_name}' if len(add_name) > 0 else ''
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        postfix += f'_{timestamp}'

        return postfix

    def plot_enstrophy_spectrum(self,
                                transect_name='along_flow',
                                data={},
                                ):

        S_truth = self.ct.compute_spectrum_along_transect(
            data['truth'],
            data['scaler_truth'],
            transect_name=transect_name,
            spectrum_type='enstrophy')
        S_pred = self.ct.compute_spectrum_along_transect(
            data['pred'],
            data['scaler_truth'],
            transect_name=transect_name,
            spectrum_type='enstrophy')
        S_lowres = self.ct.compute_spectrum_along_transect(
            data['lowres'],
            data['scaler_lowres'],
            transect_name=transect_name,
            spectrum_type='enstrophy')

        # compute mean
        S_truth_mn = np.mean(S_truth, axis=0)
        S_pred_mn = np.mean(S_pred, axis=0)
        S_lowres_mn = np.mean(S_lowres, axis=0)

        plt.figure()
        plt.loglog(S_truth_mn, '.-', label='HR truth')
        plt.loglog(S_pred_mn, '.-', label='Model prediction')
        plt.loglog(S_lowres_mn, '.-', label='LR forcing/control')
        plt.legend()
        plt.grid()
        plt.gca().set_ylim([1e-5, 1])
        plt.gca().set_title(f'Mean eddy enstrophy spectrum, {transect_name}')

        postfix = self.create_postfix()
        fig_name = (f'{self.results_dir}/'
                    f'enstrophy_spectrum_{transect_name}{postfix}.png')
        print(fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)

    def plot_energy_spectrum(self,
                             transect_name='along_flow',
                             data={},
                             ):

        S_truth = self.ct.compute_spectrum_along_transect(
            data['truth'],
            data['scaler_truth'],
            transect_name=transect_name,
            spectrum_type='energy')

        S_pred = self.ct.compute_spectrum_along_transect(
            data['pred'],
            data['scaler_truth'],
            transect_name=transect_name,
            spectrum_type='energy')

        S_lowres = self.ct.compute_spectrum_along_transect(
            data['lowres'],
            data['scaler_lowres'],
            transect_name=transect_name,
            spectrum_type='energy')

        # compute mean
        S_truth_mn = np.mean(S_truth, axis=0)
        S_pred_mn = np.mean(S_pred, axis=0)
        S_lowres_mn = np.mean(S_lowres, axis=0)

        k_1 = np.linspace(1.7, np.ceil(len(S_truth_mn) / 2), 100)
        k_2 = np.linspace(7, len(S_truth_mn), 100)

        offset_1 = 1e1 * np.max(S_truth_mn) if transect_name == 'along_flow'\
            else 1e0 * np.max(S_truth_mn)
        offset_2 = 2e2 * np.max(S_truth_mn) if transect_name == 'along_flow'\
            else 1e1 * np.max(S_truth_mn)

        plt.figure()
        plt.loglog(S_truth_mn, '.-', label='HR truth')
        plt.loglog(S_pred_mn, '.-', label='Model prediction')
        plt.loglog(S_lowres_mn, '.-', label='LR forcing/control')
        plt.loglog(k_1, offset_1 * k_1**(-5 / 3), '--', label='k^-5/3')
        plt.loglog(k_2, offset_2 * k_2**(-3), ':', label='k^-3')
        plt.legend()
        plt.gca().set_ylim([1e-7, 1])
        plt.gca().set_title(
            f'Mean eddy kinetic energy spectrum, {transect_name}'
        )
        plt.grid()

        postfix = self.create_postfix()
        fig_name = (f'{self.results_dir}/energy_spectrum_{transect_name}'
                    f'{postfix}.png'
                    )
        print(fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)

        return {'truth': S_truth,
                'lowres': S_lowres,
                'pred': S_pred}

    def unscale2D(self, field, scaler):
        x, y = field.shape
        field = scaler\
            .inverse_transform(field.reshape(1, -1))\
            .reshape(x, y)
        return field

    def plot_swot_spectrum(self, data={}):
        """Compute spectrum along a transect. To avoid aliasing problems we
        regrid the field to a grid with a diagonal that matches the
        transect.

        """
        print("Plotting SWOT spectra and more")

        # get the grid
        grid_orig = data['grid']
        lons = grid_orig['lon'][0, :]
        lats = grid_orig['lat'][:, 0]

        test_range = data['test_range']
        N_test = len(test_range)
        N_plots = 5
        field_indexes = np.arange(0, N_test, np.ceil(N_test / N_plots))\
            .astype(int)

        def regrid_and_interp(field, regridder):
            field_regr = regridder(np.ascontiguousarray(field))
            line_values = np.diag(field_regr)
            nanmask = ~np.isnan(line_values)
            x = np.arange(len(nanmask))
            line_values = np.interp(x, x[nanmask], line_values[nanmask])
            return line_values

        plt.close('all')
        plt.figure(figsize=self.figsize)
        for plot_i, index in enumerate(field_indexes):

            index_global = test_range[index]
            start = data['transects']['start'][index_global]
            end = data['transects']['end'][index_global]
            length = data['transects']['length'][index_global]

            transect = {
                'lon_start': lons[start[0]],
                'lon_end': lons[end[0]],
                'lat_start': lats[start[1]],
                'lat_end': lats[end[1]]
            }

            regridder = tools.regrid_to_transect(grid_orig,
                                                 resolution=length,
                                                 **transect)

            scaler = data['scaler_lowres']
            # HR: high-res SWOT
            HR_field = self.unscale2D(data['truth'][index,].squeeze(), scaler)
            # LR: low-res original
            LR_field = self.unscale2D(data['lowres'][index,].squeeze(), scaler)
            # PR: predicted reconstruction
            PR_field = self.unscale2D(data['pred'][index,].squeeze(), scaler)

            HR_values = regrid_and_interp(HR_field, regridder)
            LR_values = regrid_and_interp(LR_field, regridder)
            PR_values = regrid_and_interp(PR_field, regridder)

            HR_k, HR_A, _ = line_power_spectrum(HR_values)
            LR_k, LR_A, _ = line_power_spectrum(LR_values)
            PR_k, PR_A, _ = line_power_spectrum(PR_values)

            ax1 = plt.subplot(3, N_plots, plot_i + 1)
            ax1.loglog(HR_k, HR_A, 'k-', label='HR: high-res SWOT')
            ax1.loglog(LR_k, LR_A, label='LR: low-res original')
            ax1.loglog(PR_k, PR_A, label='PR: predicted')
            ax1.loglog(HR_k[10:-30], 1e3 * HR_k[10:-30]**(-5 / 3),
                       '--', label='k^-5/3')
            ax1.loglog(HR_k[10:-30], 1e4 * HR_k[10:-30]**(-3),
                       '--', label='k^-3')

            ax1.grid(True, which="both", linestyle='--',
                     linewidth=0.5, alpha=0.7)
            if plot_i == 0:
                ax1.legend()
            ax1.set_title(f"{np.datetime64(data['time'][index].data, 'D')}")

            ax2 = plt.subplot(3, N_plots, N_plots + plot_i + 1)
            ax2.pcolormesh(LR_field)
            ax2.pcolormesh(HR_field)
            ax2.contour(np.isnan(HR_field),
                        levels=1, colors='k',
                        linewidths=0.4)
            ax2.invert_yaxis()
            rr, cc = line(start[0], start[1], end[0], end[1])
            ax2.plot(rr, cc, 'r--', linewidth=2)
            ax2.axis('off')

            ax3 = plt.subplot(3, N_plots, 2 * N_plots + plot_i + 1)
            ax3.pcolormesh(PR_field)
            # ax3.plot(rr, cc, 'r--', linewidth=2)
            ax3.axis('off')
            ax3.invert_yaxis()
            ax3.set_title('predicted')

        postfix = self.create_postfix()
        fig_name = (f'{self.results_dir}/spectra_swot'
                    f'{postfix}.png'
                    )

        print(fig_name)
        plt.savefig(fig_name, bbox_inches='tight')


def line_power_spectrum(line_values):

    fourier_line = np.fft.fft(line_values)
    power_spectrum = np.abs(fourier_line) ** 2

    n = len(line_values)
    kfreq = np.fft.fftfreq(n) * n
    kfreq = np.abs(kfreq)

    kbins = np.arange(0.5, n / 2, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = binned_statistic(kfreq, power_spectrum,
                                   statistic="mean", bins=kbins)
    return kvals, Abins, line_values


def generate_transect(field, x_res=10, early_stop=250):
    """Tries to find a straight line through as much non-nan values in
     <field> as possible.

     Returns start and end indices.

    """
    ny, nx = field.shape

    # try a number of preset start_x values
    start_x_range = np.arange(1, int(nx / 2),
                              step=int(nx / x_res / 2))

    # get a list of non-nan indices
    indices = np.where(~np.isnan(field))

    length = 0
    best_startpoint = []
    best_endpoint = []

    # iterate over selected start_x
    for start_x in start_x_range:
        # nonnan indices in column
        nonnans = np.where(~np.isnan(field[:, start_x]))[0]

        # discard narrow columns
        if len(nonnans) < 5:
            continue

        # get the starting point halfway the first chunk of nonnans
        diff = nonnans[1:] - nonnans[:-1]
        ind = np.where(diff > 1)
        if len(ind[0]) > 0:
            nonnans = nonnans[:ind[0][0] + 1]
        pad = int(np.round(len(nonnans) / 2))
        start_y = nonnans[pad]

        # full start index
        start = [start_x, start_y]
        # iterate over all non-nans and find longest line
        for i, j in zip(indices[0], indices[1]):
            end = [j, i]
            rr, cc = line(start[0], start[1], end[0], end[1])
            line_values = field[cc, rr]

            # check line + neighbourhood for nans
            try:
                line_extended = np.concatenate([line_values,
                                                field[cc - 1, rr],
                                                field[cc + 1, rr],
                                                field[cc, rr - 1],
                                                field[cc, rr + 1]])
            except IndexError:
                continue

            if np.any(np.isnan(line_extended)):
                continue
            elif len(line_values) > length:
                length = len(line_values)
                best_endpoint = end
                best_startpoint = start

        if length > early_stop:
            break

    # finalize
    print(f'length transect: {length}')
    end = best_endpoint
    start = best_startpoint

    return start, end, length
