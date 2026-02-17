from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import os
import cmocean
import tools
from multiprocess import Pool
import importlib
from skimage.draw import line
from scipy.stats import binned_statistic
import compute_tool

importlib.reload(compute_tool)


class PlotMachine():
    def __init__(self,
                 dm,
                 output_dict={},
                 time_array=None,
                 trial_id=None,
                 figsize=(16, 8),
                 ):

        self.dm = dm
        self.dirs = dm.dirs
        self.log_dir = f"{self.dirs['logs']}"
        self.results_dir = self.dirs['results']
        self.figsize = figsize
        self.output_dict = output_dict
        self.time_array = time_array
        self.cbar_shrinkf = 0.5
        self.frame_stride = 4
        self.pool_size = 1
        self.trial_id = trial_id
        self.postfix = ''

        # create compute tool object
        self.ct = compute_tool.ComputeTool(dm=self.dm)

    def create_results_dir(self, epoch):
        self.results_dir = f"{self.dirs['results']}/epoch{epoch}{self.postfix}"
        os.system(f'mkdir -p {self.results_dir}')

    def set_results_dir(self, dirname):
        self.results_dir = dirname
        os.system(f'mkdir -p {self.results_dir}')

    def plot_reconstructions(self, plot_dict):
        plt.close('all')
        metadata = plot_dict['meta']
        plot_dict.pop('meta', None)
        prefix = metadata['prefix']
        fig_name = (f"{self.results_dir}/"
                    f"{prefix}reconstructions_t{metadata['time']:04d}.png")

        num_plots = len(plot_dict.keys())
        M, N = metadata['subplot_shape']
        while M * N < num_plots:
            M += 1

        plt.figure(figsize=(N * 7, M * 4))
        for i, (key, item) in enumerate(plot_dict.items()):
            plt.subplot(M, N, i + 1)
            a = plt.pcolormesh(item['data'],
                               vmin=item['vmin'],
                               vmax=item['vmax'],
                               )
            plt.colorbar(a)
            plt.gca().set_title(key)

        print(fig_name)
        plt.savefig(fig_name, bbox_inches='tight')

    def plot_coarse_input(
            self,
            data_dict,
            target_date='2025-05-01',
            field_type='uo'
    ):
        plt.close('all')
        ensembles, normal_runs = tools.split_ensembles(data_dict.keys())
        time = data_dict[normal_runs[0]]['time']
        t_idx = np.where(time > np.datetime64(target_date))[0][0]

        for run in normal_runs:
            if 'bilin' not in run:
                continue

            values = data_dict[run]['input']['LR_input_orig'][t_idx, ]
            grid_LR = data_dict[run]['input']['LR_grid']
            lons = grid_LR['lon']
            lats = grid_LR['lat']

            if field_type == 'uo':
                values = values[..., 0]
                # label = 'zonal velocity ($m/s$)'
                cmap = 'RdBu'
                vmin = -0.6
                vmax = 0.6

            if field_type == 'vo':
                values = values[..., 1]
                # label = 'zonal velocity ($m/s$)'
                cmap = 'RdBu'
                vmin = -0.6
                vmax = 0.6

            elif field_type == 'energy':
                values = np.sum(np.square(values[..., :2]), axis=-1)
                # label = 'kinetic energy ($m^2/s^2$)'
                cmap = 'coolwarm'
                vmin = None
                vmax = None

            plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.pcolormesh(lons, lats,
                          values,
                          transform=ccrs.PlateCarree(),
                          shading='auto',
                          cmap=cmap,
                          vmin=vmin,
                          vmax=vmax)
            ax.coastlines(resolution='10m', color='black',
                          linewidth=1)

            ax.pcolormesh(lons, lats, np.zeros_like(lons[:-1, :-1]),
                          transform=ccrs.PlateCarree(),
                          shading='flat',
                          facecolor='none',
                          edgecolor='black',
                          alpha=0.6,
                          linewidth=0.5)

            cf = data_dict[run]['cf']

            ax.set_extent([lons[0, 0], lons[-1, -1],
                           lats[0, 0], lats[-1, -1]],
                          crs=ccrs.PlateCarree())

            sizes = {8: 10, 16: 30, 32: 45}
            ax.scatter(lons, lats, c='k',
                       s=sizes[cf],
                       zorder=10, alpha=1,
                       clip_on=False)

            runid = run.split('/')[0]
            fig_name = (f'{self.results_dir}/'
                        f'{runid}_LRinput_{field_type}_'
                        f'{target_date}.png')
            print(fig_name)
            plt.savefig(fig_name, bbox_inches='tight', dpi=200)

    def plot_2d_fields(
            self,
            data_dict,
            target_date='2025-05-01',
            field_type='vorticity',
            overview=False,
            add_contours=False
    ):

        plt.close('all')
        ensembles, normal_runs = tools.split_ensembles(data_dict.keys())
        # field_type = 'vorticity'
        # get time index
        # target_date = '2025-06-01'
        time = data_dict[normal_runs[0]]['time']
        t_idx = np.where(time > np.datetime64(target_date))[0][0]

        if field_type == 'vorticity':
            lons = self.dm.grid_HR['lon'][1:, 1:]
            lats = self.dm.grid_HR['lat'][1:, 1:]
        else:
            lons = self.dm.grid_HR['lon']
            lats = self.dm.grid_HR['lat']

        lon_min = np.min(lons[0, 0])
        lon_max = np.min(lons[-1, -1])
        lat_min = np.min(lats[0, 0])
        lat_max = np.min(lats[-1, -1])

        if overview:
            normal_runs = ['truth']
            ensembles = {}
            add_contours = False

        all_runs = normal_runs + [value[-1]
                                  for value in ensembles.values()]
        for run in all_runs:
            values = data_dict[run]['data'][t_idx, ]

            if field_type == 'vorticity':
                values = self.ct.vorticity(np.expand_dims(values, 0), None)
                label = 'vorticity (cycles/day)'
                cmap = 'RdBu'
                vmin = -12
                vmax = 12
                extend = 'both'

            elif field_type == 'ssh':
                values = values[..., -1]
                label = 'ssh ($m$)'
                cmap = 'RdBu'
                vmin = None
                vmax = None
                extend = None

            elif field_type == 'energy':
                values = np.sum(np.square(values[..., :2]), axis=-1)
                label = 'kinetic energy ($m^2/s^2$)'
                cmap = 'coolwarm'
                vmin = 0
                vmax = 0.6
                extend = 'max'

            elif field_type == 'uo':
                values = values[..., 0]
                label = 'zonal velocity ($m/s$)'
                cmap = 'RdBu'
                vmin = -0.6
                vmax = 0.6
                extend = 'both'

            elif field_type == 'vo':
                values = values[..., 1]
                label = 'meridional velocity ($m/s$)'
                cmap = 'RdBu'
                vmin = -0.6
                vmax = 0.6
                extend = 'both'

            mask = self.dm.mask.values[0,]

            # masking
            if mask.shape == values.shape:
                values = np.where(self.dm.mask.values[0,] == 0, np.nan, values)
            else:
                values = np.where(values == 0.0, np.nan, values)

            plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())
            mesh = ax.pcolormesh(lons, lats,
                                 values,
                                 transform=ccrs.PlateCarree(),
                                 cmap=cmap,
                                 vmin=vmin,
                                 vmax=vmax)

            if add_contours:
                ax.contour(lons, lats,
                           values,
                           colors='k',
                           linewidth=1,
                           corner_mask=False,
                           antialiased=True,
                           transform=ccrs.PlateCarree(),
                           levels=12, alpha=0.3)

            ax.coastlines(resolution='10m', color='black',
                          linewidth=1)

            if overview:
                ax.set_xlim([-4, 11])
                ax.set_ylim([52, 62])
                ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
                        [lat_min, lat_min, lat_max, lat_max, lat_min],
                        'k--',
                        transform=ccrs.PlateCarree())
                ax.gridlines(draw_labels=True,
                             linestyle='--', color='gray', alpha=0.3)

            else:
                plt.colorbar(mesh, orientation='horizontal',
                             label=label,
                             pad=0.05,
                             extend=extend)

            if run == 'truth' and not overview:
                _, tr_along = self.ct.get_transect('along_flow')
                _, tr_across = self.ct.get_transect('across_flow')
                ax.plot([tr_along['lon_start'], tr_along['lon_end']],
                        [tr_along['lat_start'], tr_along['lat_end']],
                        'k-',
                        linewidth=2.6,
                        transform=ccrs.PlateCarree(),
                        label='along-flow transect',
                        )
                ax.plot([tr_across['lon_start'], tr_across['lon_end']],
                        [tr_across['lat_start'], tr_across['lat_end']],
                        'k--',
                        linewidth=2.6,
                        transform=ccrs.PlateCarree(),
                        label='across-flow transect',
                        )
                ax.legend(loc='lower left', fontsize='large')
                fig_name = (f'{self.results_dir}/'
                            f'reference_{field_type}_'
                            f'transects_{target_date}.png')
                plt.savefig(fig_name, bbox_inches='tight', dpi=200)

            elif not overview:
                runid = run.split('/')[0]
                fig_name = (f'{self.results_dir}/'
                            f'{runid}_{field_type}_{target_date}.png')
                print(fig_name)
                plt.savefig(fig_name, bbox_inches='tight', dpi=200)

        # export overview
        if overview:
            fig_name = (f'{self.results_dir}/overview_'
                        f'{field_type}_{target_date}.png'
                        )
            print(fig_name)
            plt.savefig(fig_name, bbox_inches='tight', dpi=200)

    def plot_hovmöller(
            self,
            data_dict,
            compute=False,
            scaler=None,
            plot_type='vorticity',
            transect='along_flow',
    ):

        if compute:
            d = {}
            for key, value in data_dict.items():
                data = value['data']
                d[key] = self.ct.hovmöller_along_transect(
                    data,
                    scaler=None,
                    transect_name=transect,
                    spectrum_type=plot_type,
                    detide=False,
                )
            T = d
        else:
            T = data_dict

        for key, value in T.items():
            if plot_type == 'energy':
                # take only first 2 variables uo, vo
                T[key] = 0.5 * np.sum(np.square(value[..., :2]), axis=2)
            elif plot_type == 'uo':
                T[key] = value[..., 0]
            elif plot_type == 'enstrophy':
                T[key] = np.square(value / (self.ct.tdim / 3600))

        vmin = None
        vmax = None
        cmap = 'viridis'
        label = None
        extend = None

        if plot_type == 'vorticity':
            cmap = 'RdBu'
            vmin = -12
            vmax = 12
            label = 'vorticity (cycles/day)'
            extend = 'both'

        if plot_type == 'enstrophy':
            cmap = 'YlGnBu'
            cmap = cmocean.cm.tempo
            vmin = 0
            vmax = 0.7
            label = 'enstrophy ($h^{-2}$)'
            extend = 'max'

        if plot_type == 'uo':
            cmap = 'RdBu'
            vmin = -0.6
            vmax = 0.6
            label = 'horizontal velocity ($m/s$)'
            extend = 'both'

        if plot_type == 'energy':
            cmap = 'YlGnBu'
            cmap = cmocean.cm.tempo
            vmin = 0
            vmax = 0.7
            label = 'energy $(m/s)^2$'
            extend = 'max'

        if 'time' in data_dict['truth']:
            t_array = data_dict['truth']['time']
        else:
            t_array = np.arange(T['truth'].shape[0])

        x_array = np.arange(T['truth'].shape[1])

        for i, key in enumerate(T.keys()):
            plt.figure(figsize=(7, 2))
            a = plt.pcolormesh(
                t_array,
                x_array,
                T[key].transpose(),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap)

            plt.colorbar(a, label=label, aspect=10, extend=extend)
            plt.xticks(rotation=45, ha='right')
            plt.yticks([])
            # plt.ylabel('distance along transect')
            runid = key.split('/')[0]
            fig_name = (f'{self.results_dir}/Hovmöller_'
                        f'{runid}_{plot_type}_{transect}.png')
            print(fig_name)
            plt.savefig(fig_name, bbox_inches='tight', dpi=200)

    def spectra_wrapper(self, x, y, z, t):
        scaler_list = ['LR', 'HR', 'HR']
        if x.shape == z.shape:
            scaler_list[0] = 'HR'

        x_unscaled, y_unscaled, z_unscaled = \
            [tools.unscale_var(d, self.dm.scalers[res])
             for d, res in zip([x, y, z], scaler_list)]

        if x_unscaled.shape != z_unscaled.shape:
            # upsample unscaled x (bilinear interpolation)
            x_unscaled = \
                np.ascontiguousarray(x_unscaled.transpose((0, 3, 1, 2)))
            x_unscaled = \
                self.dm\
                    .bilin_upsampler(x_unscaled)\
                    .transpose((0, 2, 3, 1))

        cmap = plt.get_cmap('tab10')
        data = {
            'lowres': {
                'data': np.nan_to_num(x_unscaled),
                'time': [],
                'plotkwargs': {
                    'label': 'bilinear interpolation',
                    'linestyle': '--',
                    'color': cmap(5),
                    'zorder': 0,
                },
            },
            'truth': {
                'data': np.nan_to_num(y_unscaled),
                'time': [],
                'plotkwargs': {
                    'label': 'high-resolution truth',
                    'linestyle': '-',
                    'color': cmap(0),
                    'zorder': 10,
                },
            },
            'pred': {
                'data': np.nan_to_num(z_unscaled),
                'time': [],
                'plotkwargs': {
                    'label': 'prediction',
                    'linestyle': '-',
                    'color': cmap(1),
                    'zorder': 4,
                },
            },
        }

        # do not create temporal plots when time dimension is limited
        # (during testing mainly)
        Nt = x.shape[0]
        directions = ['spatial', 'temporal'] if Nt > 100 else ['spatial']

        for transect in ['along_flow', 'across_flow']:
            for spectype in ['energy', 'enstrophy']:
                for direction in directions:
                    S, T = self.plot_spectrum(data,
                                              transect_name=transect,
                                              spectrum_type=spectype,
                                              direction=direction)
                self.plot_hovmöller(
                    T,
                    plot_type=spectype,
                    transect=transect,
                )

    def plot_single_frame(self, frame_id, output_dict=None):
        self.output_dict = self.output_dict \
            if output_dict is None else output_dict

        plt.figure(figsize=self.figsize)
        fig_name = f'{self.results_dir}/results_autoencoder{self.postfix}.png'
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

        movie_name = f'movie{self.postfix}.mov'
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

    def plot_history(self,
                     hist,
                     ):

        if not isinstance(hist, dict):
            history = hist.history
            fig_name = f'{self.log_dir}/history{self.postfix}.png'
        else:
            history = hist
            fig_name = f'{self.log_dir}/history_intermediate.png'

        plt.close('all')
        plt.figure(figsize=(11, 9))
        [plt.semilogy(value, label=key) for key, value in history.items()]
        plt.grid(which='both')
        plt.gca().set_xlabel('epoch')
        plt.legend()

        print('\n', fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)

    def plot_prediction_error(self, X, Y, Z, add_name=''):

        fig_name = f'{self.results_dir}/errors{self.postfix}.png'

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

        self.postfix = ''
        if self.trial_id is not None:
            self.postfix += f'_trial_{self.trial_id}'

        self.postfix += f'_{add_name}' if len(add_name) > 0 else ''
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.postfix += f'_{timestamp}'

    def plot_timestepping(self,
                          results,
                          truths,
                          epoch,
                          masking,
                          add_name='',
                          ):
        plt.close('all')
        results_HR = np.concatenate([r['HR_data'] for r in results], 0)
        truths_HR = np.concatenate([t['HR_data'] for t in truths], 0)
        time_arr = [r['time'] for r in results]

        result_norms = [np.linalg.norm(t[0,
                                         masking['rows'],
                                         masking['cols'],])
                        for t in results_HR]

        truth_norms = [np.linalg.norm(t[0,
                                        masking['rows'],
                                        masking['cols'],])
                       for t in truths_HR]

        plt.figure(figsize=self.figsize)
        plt.plot(time_arr, result_norms, '.-', label='timestepping')
        plt.plot(time_arr, truth_norms, '.-', label='truths')
        plt.legend()
        plt.grid(which='both')
        plt.gca().tick_params(axis='x', labelrotation=45)
        plt.tight_layout()

        fig_name = \
            f'{self.results_dir}/timestepping.png'
        print(fig_name)
        plt.savefig(fig_name)

        # plot latent variables in time
        if 'ls_mean' in results[0]:
            ls = np.concatenate([r['ls_mean'] for r in results], 0)
        else:
            ls = []

        if len(ls) > 0:
            ls = np.reshape(ls, (ls.shape[0], -1))
            ls = ls.transpose()

            plt.figure(figsize=self.figsize)
            plt.subplot(2, 1, 1)
            a = plt.imshow(ls, aspect='auto', interpolation=None)
            plt.colorbar(a)
            plt.subplot(2, 1, 2)
            rr = np.min([32, ls.shape[0]])
            a = plt.imshow(ls[:rr,], aspect='auto', interpolation=None)
            plt.colorbar(a)
            plt.suptitle('latent space plots')
            plt.tight_layout()
            fig_name = \
                f'{self.results_dir}/ls_mean.png'
            print(fig_name)
            plt.savefig(fig_name, bbox_inches='tight')
        else:
            print('no latent variables available for plotting')

        # plot latent variables in time
        if 'ls_pred' in results[0]:
            ls = np.concatenate([r['ls_pred'] for r in results], 0)
        else:
            ls = []

        if len(ls) > 0:
            ls = np.reshape(ls, (ls.shape[0], -1))
            ls = ls.transpose()

            plt.figure(figsize=self.figsize)
            plt.subplot(2, 1, 1)
            a = plt.imshow(ls, aspect='auto', interpolation=None)
            plt.colorbar(a)
            plt.subplot(2, 1, 2)
            rr = np.min([32, ls.shape[0]])
            a = plt.imshow(ls[:rr,], aspect='auto', interpolation=None)
            plt.colorbar(a)
            plt.suptitle('latent space plots')
            plt.tight_layout()
            fig_name = \
                f'{self.results_dir}/ls_pred.png'
            print(fig_name)
            plt.savefig(fig_name, bbox_inches='tight')
        else:
            print('no latent variables available for plotting')

    def plot_spectrum(self,
                      data,
                      transect_name='along_flow',
                      spectrum_type='energy',
                      direction='spatial',
                      add_powerlaws=False,
                      detide=False,
                      combine_members='quantiles',
                      make_title=False,
                      plot_legend=True,
                      ):
        k = {}
        S = {}
        T = {}
        for key, value in data.items():
            k[key], S[key], T[key] = \
                self.ct.compute_spectrum_along_transect(
                    value['data'],
                    transect_name=transect_name,
                    spectrum_type=spectrum_type,
                    direction=direction,
                    detide=detide)

        # compute mean over space or time
        S_mn = {key: np.mean(value, axis=-1) for key, value in S.items()}
        S_stacked, S_combined, normal_runs = self.combine_ensemble_members(
            S_mn,
            k,
            data,
            combine_members,
        )

        # first do the normal runs (not ensembles)
        for key in normal_runs:
            plt.loglog(
                k[key],
                S_mn[key],
                **data[key]['plotkwargs'],
            )

        cf = data[key]['cf']

        # plot ensemble runs (quantiles)
        for key, value in S_combined.items():
            # plot q1,q2,q3
            plt.loglog(k[key],
                       value['q2'],
                       **data[key]['plotkwargs'])
            plotkwargs = data[key]['plotkwargs'].copy()
            data.pop(key)
            plotkwargs.pop('label')
            plotkwargs.pop('zorder')
            plt.fill_between(
                k[key],
                value['q1'],
                value['q3'],
                alpha=0.5,
                zorder=0,
                **plotkwargs,
            )

        if 'truth' in S_mn.keys():
            ymax = np.ceil(np.log10(np.max(S_mn['truth'])))
            ymin = np.floor(np.log10(np.min(S_mn['truth'])))
            plt.gca().set_ylim([10**ymin, 10**ymax])

        if add_powerlaws:
            ks = k['truth']
            plt.loglog(ks, 1e-4 * ks**(-3), ':', label='$k^{-3}$')
            plt.loglog(ks, 1e-4 * ks**(-4), '--', label='$k^{-4}$')
            plt.loglog(ks, 1e-4 * ks**(-5), '--', label='$k^{-5}$')

        postfix = f'{transect_name}_{direction}_cf{cf}.png'

        if spectrum_type == 'energy':
            tstring = \
                (f'Mean kinetic energy spectrum,'
                 f' {transect_name}, {direction}')
            fig_name = \
                (
                    f'{self.results_dir}/'
                    f'energy_spectrum_{postfix}'
                )
        elif spectrum_type == 'enstrophy':
            tstring = \
                (f'Mean enstrophy spectrum,'
                 f' {transect_name} {direction}')
            fig_name = \
                (f'{self.results_dir}/'
                 f'enstrophy_spectrum_'
                 f'{postfix}')

        elif spectrum_type == 'ssh':
            tstring = \
                (f'Mean ssh spectrum,'
                 f' {transect_name} {direction}')
            fig_name = \
                (f'{self.results_dir}/'
                 f'ssh_spectrum_'
                 f'{postfix}')
        elif spectrum_type == 'TKE':
            tstring = \
                (f'TKE spectrum,'
                 f' {transect_name} {direction}')
            fig_name = \
                (f'{self.results_dir}/'
                 f'TKE_spectrum_'
                 f'{postfix}')
        elif spectrum_type == 'MKE':
            tstring = \
                (f'MKE spectrum,'
                 f' {transect_name} {direction}')
            fig_name = \
                (f'{self.results_dir}/'
                 f'MKE_spectrum_'
                 f'{postfix}')
        else:
            raise Exception('unknown spectrum type')

        # plt.gca().set_ylim([1e-8, 1e2])
        if make_title:
            plt.gca().set_title(tstring)
        if plot_legend:
            plt.legend(loc='lower left', fontsize='large')

        plt.grid()

        print(fig_name)
        plt.tight_layout()
        plt.savefig(fig_name, dpi=200)

        return S, T

    def combine_ensemble_members(self, S, k, data, combine_members):
        # combine ensemble members
        if combine_members == 'disabled':
            return {}, {}

        ensembles, normal_runs = tools.split_ensembles(S.keys())
        S_stacked = {}
        S_combined = {}

        for key, members in ensembles.items():
            # stack ensemble members in new 0th dimension
            S_stacked.update({key: np.stack([S[mem] for mem in members])})

            S_combined[key] = {}

            # perform some operation on the data
            if combine_members == 'mean_std':
                S_combined[key]['mean'] = np.mean(S_stacked[key], 0)
                S_combined[key]['std'] = np.std(S_stacked[key], 0)

            elif combine_members == 'quantiles':
                S_combined[key]['q1'] = \
                    np.quantile(S_stacked[key], 0.25, axis=0)
                S_combined[key]['q2'] = \
                    np.quantile(S_stacked[key], 0.50, axis=0)
                S_combined[key]['q3'] = \
                    np.quantile(S_stacked[key], 0.75, axis=0)

            # replace metadata
            k.update({key: k[members[0]]})
            data.update({key: data[members[0]]})

        return S_stacked, S_combined, normal_runs

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

        fig_name = (f'{self.results_dir}/spectra_swot'
                    f'{self.postfix}.png'
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
