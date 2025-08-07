from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocess import Pool
from importlib import reload
import data_utils
reload(data_utils)
from data_utils import DataFactory
import compute_tool
reload(compute_tool)
from compute_tool import ComputeTool


class PlotMachine():
    def __init__(self, figsize=(16, 8),
                 output_dict={},
                 time_array=None,
                 results_dir=None,
                 movie_dir=None,
                 trial_id=None):

        self.figsize = figsize
        self.output_dict = output_dict
        self.time_array = time_array
        self.results_dir = results_dir
        self.movie_dir = movie_dir
        self.cbar_shrinkf = 0.5
        self.frame_stride = 4
        self.pool_size = 1
        self.trial_id = trial_id
        self.dm = DataFactory()

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

        RSE_Y = np.sqrt(np.sum(np.square(X-Y),axis=(1,2,3)))
        RSE_Z = np.sqrt(np.sum(np.square(X-Z),axis=(1,2,3)))

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
        if self.trial_id != None:
            postfix += f'_trial_{self.trial_id}'

        postfix += f'_{add_name}' if len(add_name)>0 else ''
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        postfix += f'_{timestamp}'

        return postfix


    def plot_enstrophy_spectrum(self,
                                transect_name='along_flow',
                                data = {},
                                add_coarse_data=False):

        # get coarse data:
        if add_coarse_data:
            do = self.dm.get_coarse_data(data['time'])

        ct = ComputeTool()
        S_truth  = ct.compute_spectrum_along_transect(
            data['truth'],
            data['scaler_truth'],
            transect_name=transect_name,
            spectrum_type='enstrophy')
        S_pred  = ct.compute_spectrum_along_transect(
            data['pred'],
            data['scaler_truth'],
            transect_name=transect_name,
            spectrum_type='enstrophy')
        S_lowres  = ct.compute_spectrum_along_transect(
            data['lowres'],
            data['scaler_lowres'],
            transect_name=transect_name,
            spectrum_type='enstrophy')

        if add_coarse_data:
            S_coarse  = ct.compute_spectrum_along_transect(
                do,
                None,
                transect_name=transect_name,
                spectrum_type='energy')

        # compute mean
        S_truth_mn  = np.mean(S_truth, axis=0)
        S_pred_mn   = np.mean(S_pred, axis=0)
        S_lowres_mn = np.mean(S_lowres, axis=0)
        if add_coarse_data:
            S_coarse_mn = np.mean(S_coarse, axis=0)

        n = len(S_truth_mn)
        kvals = np.arange(1,n+1)

        plt.figure()
        plt.loglog(S_truth_mn, '.-', label='HR truth')
        plt.loglog(S_pred_mn, '.-', label='Model prediction')
        plt.loglog(S_lowres_mn, '.-', label='LR forcing/control')
        if add_coarse_data:
            plt.loglog(S_coarse_mn, '.-', label='Coarse model')
        plt.legend()
        plt.grid()
        plt.gca().set_ylim([1e-5,1])
        plt.gca().set_title(f'Mean eddy enstrophy spectrum, {transect_name}')

        postfix = self.create_postfix()
        fig_name = (f'{self.results_dir}/'
                    f'enstrophy_spectrum_{transect_name}{postfix}.png')
        print(fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)

    def plot_energy_spectrum(self,
                             transect_name='along_flow',
                             data = {},
                             add_coarse_data=False):

        if add_coarse_data:
            do = self.dm.get_coarse_data(data['time'])

        ct = ComputeTool()
        S_truth  = ct.compute_spectrum_along_transect(
            data['truth'],
            data['scaler_truth'],
            transect_name=transect_name,
            spectrum_type='energy')

        S_pred  = ct.compute_spectrum_along_transect(
            data['pred'],
            data['scaler_truth'],
            transect_name=transect_name,
            spectrum_type='energy')

        S_lowres  = ct.compute_spectrum_along_transect(
            data['lowres'],
            data['scaler_lowres'],
            transect_name=transect_name,
            spectrum_type='energy')

        if add_coarse_data:
            S_coarse  = ct.compute_spectrum_along_transect(
                do,
                None,
                transect_name=transect_name,
                spectrum_type='energy')

        # compute mean
        S_truth_mn  = np.mean(S_truth, axis=0)
        S_pred_mn   = np.mean(S_pred, axis=0)
        S_lowres_mn = np.mean(S_lowres, axis=0)
        if add_coarse_data:
            S_coarse_mn = np.mean(S_coarse, axis=0)

        k_1 = np.linspace(1.7,np.ceil(len(S_truth_mn)/2), 100)
        k_2 = np.linspace(7,len(S_truth_mn), 100)

        offset_1 = 1e1*np.max(S_truth_mn) if transect_name == 'along_flow'\
            else 1e0*np.max(S_truth_mn)
        offset_2 = 2e2*np.max(S_truth_mn) if transect_name == 'along_flow'\
            else 1e1*np.max(S_truth_mn)

        plt.figure()
        plt.loglog(S_truth_mn, '.-' , label='HR truth')
        plt.loglog(S_pred_mn, '.-'  , label='Model prediction')
        plt.loglog(S_lowres_mn, '.-', label='LR forcing/control')
        if add_coarse_data:
            plt.loglog(S_coarse_mn, '.-', label='Coarse model')
        plt.loglog(k_1, offset_1 * k_1**(-5/3), '--', label='k^-5/3')
        plt.loglog(k_2, offset_2 * k_2**(-3), ':', label='k^-3')
        plt.legend()
        plt.gca().set_ylim([1e-7,1])
        plt.gca().set_title(f'Mean eddy kinetic energy spectrum, {transect_name}')
        plt.grid()

        postfix = self.create_postfix()
        fig_name = f'{self.results_dir}/energy_spectrum_{transect_name}{postfix}.png'
        print(fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)
        # plt.pause(1)

        return {'truth' : S_truth,
                'lowres' : S_lowres,
                'pred' : S_pred}
