from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocess import Pool
from transectpicker.transectpicker import TransectPicker
import dill
import data_manager as dm

class PlotMachine():

    def __init__(self, figsize=(20,12),
                 output_dict={},
                 time_array=None,
                 results_dir=None,
                 movie_dir=None,
                 trial_id=None):

        self.figsize=figsize
        self.output_dict=output_dict
        self.time_array=time_array
        self.results_dir=results_dir
        self.movie_dir=movie_dir
        self.cbar_shrinkf=0.7
        self.frame_stride=4
        self.pool_size=6
        self.trial_id=trial_id

        self.transect_dir = f'{dm.data_dir}/transects'
        os.system(f'mkdir -p {self.transect_dir}')

    def plot_single_frame(self, frame_id, output_dict=None):
        self.output_dict = self.output_dict \
            if output_dict == None else output_dict

        fig = plt.figure(figsize=self.figsize)
        postfix = self.create_postfix()
        fig_name = f'{self.results_dir}/results_autoencoder{postfix}.png'
        self.plot_frame(frame_id, fig_name)

    def create_movie(self, output_dict=None):

        self.output_dict = self.output_dict \
            if output_dict == None else output_dict

        fig = plt.figure(figsize=self.figsize)
        with Pool(self.pool_size) as p:
            p.map(self.plot_frame, range(0,len(self.time_array),
                                         self.frame_stride))

        postfix = self.create_postfix()
        movie_name = f'movie{postfix}.mov'
        framerate = 24
        sys_cmd = ( f"ffmpeg -r {framerate} -f image2 -pattern_type glob -i "
                    f"'{self.movie_dir}/frame-*.png' "
                    f"-vcodec libx264 -crf 25  -pix_fmt yuv420p -y "
                    f"{self.movie_dir}/{movie_name}" )

        print(sys_cmd)
        os.system(sys_cmd)
        sys_cmd = ( f"rm {self.movie_dir}/frame-*.png")
        print(sys_cmd)
        os.system(sys_cmd)

    def plot_frame(self, id, fig_name=None):
        plt.clf()
        if fig_name == None:
            fig_name = f'{self.movie_dir}/frame-{id:06d}.png'

        dim = int(np.ceil(np.sqrt(len(self.output_dict))))
        for f, (key, item) in enumerate(self.output_dict.items()):

            plt.subplot(dim,dim,f+1)
            h = plt.imshow(item['values'](id),
                           cmap=item['cmap'],
                           vmin=item['vmin'],
                           vmax=item['vmax'])
            plt.colorbar(h, shrink=self.cbar_shrinkf)
            plt.gca().set_title(key)
            plt.gca().invert_yaxis()

        plt.suptitle(f"date: {np.datetime64(self.time_array[id], 'h')}")
        plt.savefig(fig_name)

    def plot_history(self, hist):

        postfix = self.create_postfix()
        fig_name = f'{self.results_dir}/history{postfix}.png'

        plt.close('all')
        plt.subplot(2,1,1)
        plt.semilogy(hist.history['loss'],'.-',
                     label='loss')

        plt.grid()
        plt.legend()
        plt.gca().set_xlabel('epoch')

        if 'val_loss' in hist.history:
            plt.subplot(2,1,1)
            plt.semilogy(hist.history['val_loss'],'.-',
                         label='validation loss')

        plt.subplot(2,1,2)
        if 'error' in hist.history:
            plt.semilogy(hist.history['error'],'.-',
                         label='validation error')

        if 'base' in hist.history:
            plt.semilogy(hist.history['base'],'.-',
                         label='validation baseline')
        plt.grid()
        plt.legend()
        plt.gca().set_xlabel('epoch')

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

    def create_transect(self, output_dict):

        fig, ax  = plt.subplots(figsize=(5,4))
        field = output_dict['Kt_HR true']['values'](200)
        im = plt.pcolormesh(field)
        tpicker = TransectPicker(im, field)
        plt.show()

        transect_name = input('Give a name for the transect\n')
        dill_file = f'{self.transect_dir}/{transect_name}.dill'

        container = {'tpicker' : tpicker}

        print(f'writing to {dill_file}')
        with open(dill_file, 'wb') as file:
            dill.dump(container, file)


    def plot_spectrum(self, transect_name='along_flow', data = {}):

        dill_file = f'{self.transect_dir}/{transect_name}.dill'
        print(f'Loading transect from {dill_file}')
        with open(dill_file, 'rb') as file:
            tpicker = dill.load(file)['tpicker']

        def inverse_transform(data, scaler):
            Nt, Nlat, Nlon, num_channels = data.shape
            return scaler.inverse_transform(data.reshape(Nt,-1))\
                         .reshape(Nt, Nlat, Nlon, num_channels)

        # invert transform and restrict to transect
        truth = inverse_transform(data['truth'], data['scaler'])\
            [:,tpicker.y_trans, tpicker.x_trans,:]
        lowres = inverse_transform(data['lowres'], data['scaler'])\
            [:,tpicker.y_trans, tpicker.x_trans,:]
        pred = inverse_transform(data['pred'], data['scaler'])\
            [:,tpicker.y_trans, tpicker.x_trans,:]
