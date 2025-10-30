import os
from typing import Tuple
from abc import ABC, abstractmethod


class DataManagerBase(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def create_training_data(self) -> Tuple[dict, dict, dict, dict]:
        pass

    def setup_directories(self, experiment_id, add_id=""):
        base_dir = f'experiments/{experiment_id}{add_id}'
        models_dir = f'{base_dir}/models'
        tuning_dir = f'{base_dir}/tuning'
        results_dir = f'{base_dir}/results'
        movie_dir = f'{base_dir}/movies'
        logs_dir = f'{base_dir}/logs'
        checkpoints_dir = f'{base_dir}/checkpoints'
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
