import os
from abc import ABC, abstractmethod


class DataManagerBase(ABC):

    @abstractmethod
    def create_training_data(self):
        pass

    def get_coarse_data(self):
        pass

    def setup_directories(self, experiment_id, add_id):
        models_dir = f'experiments/{experiment_id}{add_id}/models'
        tuning_dir = f'experiments/{experiment_id}{add_id}/tuning'
        results_dir = f'experiments/{experiment_id}{add_id}/results'
        movie_dir = f'experiments/{experiment_id}{add_id}/movies'
        logs_dir = f'experiments/{experiment_id}{add_id}/logs'
        checkpoints_dir = f'experiments/{experiment_id}{add_id}/checkpoints'
        log_file = f'{logs_dir}/log.txt'

        os.system(f'mkdir -p {models_dir}')
        os.system(f'mkdir -p {tuning_dir}')
        os.system(f'mkdir -p {movie_dir}')
        os.system(f'mkdir -p {results_dir}')
        os.system(f'mkdir -p {checkpoints_dir}')
        os.system(f'mkdir -p {logs_dir}')

        dirs = {'models'      : models_dir,
                'tuning'      : tuning_dir,
                'results'     : results_dir,
                'movies'      : movie_dir,
                'checkpoints' : checkpoints_dir,
                'logs'        : logs_dir}

        files = {'log' : log_file}

        return dirs, files
