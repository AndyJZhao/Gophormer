from typing import Any

import Gophormer.functions as uf
from abc import abstractmethod, ABCMeta
from Gophormer.proj.settings import *
from Gophormer.modules.logger import Logger
from functools import partial

from argparse import ArgumentParser
import os
from types import SimpleNamespace


class ModelConfig(metaclass=ABCMeta):
    """
    Model config
    """

    def __init__(self, model):
        self.model = model
        self.exp_name = 'default'
        self.seed = 0
        self.gpus = '0'  # Use GPU as default device
        self.wandb_name = ''
        self.wandb_id = 'None'
        self.tqdm_on = True
        self.eval_freq = 3
        self.local_rank = -1

        self.git_hash = uf.get_git_hash()
        self.dataset = (d := DEFAULT_DATASET)
        self.train_percentage = DATA_INFO[d]['train_ratio']
        self.early_stop = EARLY_STOP
        self.epochs = MAX_EPOCHS
        self.birth_time = uf.get_cur_time(t_format='%m_%d-%H_%M_%S')

        # Other attributes
        self.server = 'GCR'
        self.screen_name = ''
        self._path_list = ['checkpoint_file', 'res_file']
        self._ignored_settings = ['_ignored_settings', 'tqdm_on', 'verbose', '_file_conf_list', 'logger', 'log', 'gpu',
                                  'sample_device', 'compute_dev', 'n_class', 'n_feat', 'important_paras', 'screen_name']

    def path_init(self, additional_files=[]):
        uf.mkdir_list([getattr(self, _) for _ in (self._path_list + additional_files)])

    def wandb_init(self):
        # Turn off Wandb gradients loggings
        os.environ["WANDB_WATCH"] = "false"

        wandb_settings_given = self.wandb_name != 'OFF' or self.wandb_id != 'None'
        not_parallel = self.local_rank <= 0

        if wandb_settings_given and not_parallel:
            try:
                import wandb
                from tune.settings import PROJ_NAME
                print('Starting Wandb init')
                # ! Create wandb session
                if self.wandb_id == 'None':
                    # First time running, create new wandb
                    wandb_prefix = 'DEBUG' if self.wandb_name is None else self.wandb_name
                    group_name = wandb_prefix + f'-{self.dataset:<.4s}'
                    wandb.init(project=PROJ_NAME,group=group_name, reinit=True, config=self.model_conf)
                    self.wandb_id = wandb.run.id
                else:
                    print(f'Resume from previous wandb run {self.wandb_id}')
                    wandb.init(id=self.wandb_id, resume='must', reinit=True)
                self.wandb_on = True
            except:
                return None
        else:
            os.environ["WANDB_DISABLED"] = "true"
            return None

    @property
    def f_prefix(self):
        return f"{self.model}/{self.dataset}/l{self.train_percentage:02d}seed{self.seed}Ef{self.eval_freq}-{self.model_cf_str}e{self.epochs}"

    @property
    def res_file(self):
        return f'{TEMP_RES_PATH}{self.f_prefix}.json'

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.f_prefix}.ckpt"

    @property
    @abstractmethod
    def model_cf_str(self):
        # Model config to str
        return ValueError('The model config file name must be defined')

    @property
    def model_conf(self):
        valid_types = [str, int, float, type(None)]
        judge_para = lambda p, v: p not in self._ignored_settings and p[0] != '_' and type(v) in valid_types
        # Print the model settings only.
        return {p: v for p, v in sorted(self.__dict__.items()) if judge_para(p, v)}

    @property
    def parser(self) -> ArgumentParser:
        parser = ArgumentParser("Experimental settings")
        parser.add_argument("-g", '--gpus', default='-1', type=str,
                            help='CUDA_VISIBLE_DEVICES, -1 for cpu-only mode.')
        parser.add_argument("-d", "--dataset", type=str, default=(d := DEFAULT_DATASET))
        parser.add_argument("-t", "--train_percentage", default=DATA_INFO[d]['train_ratio'], type=int)
        parser.add_argument("-e", "--early_stop", default=EARLY_STOP, type=int)
        parser.add_argument("-f", "--eval_freq", default=1, type=int)
        parser.add_argument("-v", "--verbose", default=1, type=int, help='Verbose level, higher level generates more log, -1 to shut down')
        parser.add_argument("-w", "--wandb_name", default='OFF', type=str, help='Wandb logger or not.')
        parser.add_argument("--epochs", default=MAX_EPOCHS, type=int)
        parser.add_argument("--seed", default=0, type=int)
        return parser

    def __str__(self):
        return f'{self.model} config: \n{self.model_conf}'

    def parse_args(self):
        # ! Parse defined args
        defined_args = (parser := self.parser).parse_known_args()[0]

        # ! Reinitialize config by parsed experimental args
        self.__init__(defined_args)
        default_cf = self.model_conf.items()

        # ! Parse undefined args.
        for arg, arg_val in default_cf:
            if not hasattr(defined_args, arg):
                parser.add_argument(f"--{arg}", type=type(arg_val), default=arg_val)
        return parser.parse_args()

    def _export_train_command(self):
        """Export configs to command, booleans are currently not supported"""
        train_file = f'src/{PROJ_NAME}/models/{self.model}/train.py'
        settings = ' '.join([f'--{k}={v}' for k, v in self.model_conf.items()])
        return f'python {train_file} {settings}'
