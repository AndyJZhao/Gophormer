from pathlib import Path
import socket
from types import SimpleNamespace
import os
import numpy as np


class ServerInfo:
    def __init__(self):
        self.gpu_mem, self.gpus, self.n_gpus = 0, [], 0
        try:
            import subprocess as sp

            command = "nvidia-smi --query-gpu=memory.total --format=csv"
            gpus = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            self.gpus = np.array(range(len(gpus)))
            self.n_gpus = len(gpus)
            self.gpu_mem = round(int(gpus[0].split()[0]) / 1024)
            self.sv_type = f'{self.gpu_mem}Gx{self.n_gpus}'
        except:
            print('NVIDIA-GPU not found, set to CPU.')
            self.sv_type = f'CPU'

    def __str__(self):
        return f'SERVER INFO: {self.sv_type}'


SV_INFO = ServerInfo()

PROJ_NAME = 'Gophormer'
# ! Project Path Settings

GPU_SERVER_SETTING = {'default_gpu': '0', 'mnt_dir': f'/mnt/v-jiananzhao/{PROJ_NAME}/', 'py_path': f'{str(Path.home())}/miniconda/envs/py38/bin/python'}
CPU_SERVER_SETTING = {'default_gpu': '-1', 'mnt_dir': '', 'py_path': f'python'}
SERVER_SETTINGS = {
    'GCR-TitanVx2': {'ip': '10.185.', **GPU_SERVER_SETTING},
    'ITP-P40x4': {'ip': '168.0', **GPU_SERVER_SETTING},
    'ITP-V100x8': {'ip': '10.226.', **GPU_SERVER_SETTING},
    # 'local': {'ip': '192.168.', **CPU_SERVER_SETTING},
    'mac': {'ip': '10.46.', **CPU_SERVER_SETTING},
}
SERVER = 'ITP-P40x4'
DEFAULT_GPU = SERVER_SETTINGS[SERVER]['default_gpu']
PYTHON_PATH = SERVER_SETTINGS[SERVER]['py_path']
MNT_DIR = SERVER_SETTINGS[SERVER]['mnt_dir']

# Project path: Code path, to be executed (should be local disk)
import os.path as osp

PROJ_DIR = osp.abspath(osp.dirname(__file__)).split('src')[0]
DATA_PATH = f'{PROJ_DIR}data/'

# Temp path: to be discarded
TEMP_DIR = PROJ_DIR
TEMP_PATH = f'{TEMP_DIR}temp/'
LOG_PATH = f'{TEMP_DIR}log/'

# Mount path: to be shared
MNT_DIR = f'/mnt/v-jiananzhao/{PROJ_NAME}/'
MNT_TEMP_DIR = f'{MNT_DIR}temp/'
TEMP_RES_PATH = f'{MNT_DIR}temp_results/'
RES_PATH = f'{MNT_DIR}results/'
DB_PATH = f'{MNT_DIR}exp_db/'

METRIC = 'acc'

# ! Default Settings

EARLY_STOP = 30
MAX_EPOCHS = 100
DEFAULT_DATASET = 'cora'
DATA_INFO = {
    'cora': {'train_ratio': 60},
    'citeseer': {'train_ratio': 60},
    'blogcatalog': {'train_ratio': 60},
    'dblp': {'train_ratio': 60},
    'flickr': {'train_ratio': 60},
    'pubmed': {'train_ratio': 60},
    'arxiv': {'train_ratio': 0},
}

DATASETS = list(DATA_INFO.keys())
TR_RATIO_DICT = {d: _['train_ratio'] for d, _ in DATA_INFO.items()}

GOPHORMER_DEFAULT_SETTINGS = {
    'min_eval_epochs': {
        'cora': 5,
        'citeseer': 5,
        'blogcatalog': 10,
        # 'arxiv': 40,
        'arxiv': 0,
        'dblp': 3,
        # 'flickr': 20,
        'flickr': 10,
        # 'pubmed': 15,
        'pubmed': 10,
    }

}
