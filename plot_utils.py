from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocess import Pool

class PlotMachine():

    def __init__(self, figsize=(20,12),
                 output_dict=None,
                 time_array=None,
                 results_dir=None,
                 movie_dir=None):

        self.figsize=figsize
        self.output_dict=output_dict
        self.time_array=time_array
        self.results_dir=results_dir
        self.movie_dir=movie_dir
        self.cbar_shrinkf=0.7
        self.frame_stride=4
        self.pool_size=8

    def plot_single_frame(self, frame_id):
        fig = plt.figure(figsize=self.figsize)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_name = f'{self.results_dir}/results_autoencoder_{timestamp}.png'
        self.plot_frame(frame_id, fig_name)

    def create_movie(self):
        fig = plt.figure(figsize=self.figsize)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with Pool(self.pool_size) as p:
            p.map(self.plot_frame, range(0,len(self.time_array),
                                         self.frame_stride))

        movie_name = f'movie_{timestamp}.mov'
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
        # print(fig_name)
        plt.savefig(fig_name)


    def plot_history(self, hist):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_name = f'{self.results_dir}/history_{timestamp}.png'
        plt.close('all')
        plt.semilogy(hist.history['loss'],'.-',
                     label='loss')

        if 'val_loss' in hist.history:
            plt.semilogy(hist.history['val_loss'],'.-',
                         label='validation loss')
        plt.grid()
        plt.legend()
        plt.gca().set_xlabel('epoch')
        print(fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)

    def plot_prediction_error(self, X, Y, Z):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_name = f'{self.results_dir}/errors_{timestamp}.png'

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
