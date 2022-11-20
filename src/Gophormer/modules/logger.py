# from ray import tune
import os

import wandb

import Gophormer.functions as uf
from Gophormer.functions import json_save


class Logger():
    # Logs the wandb/std

    def __init__(self, cf):
        self.cf = cf
        self._static_log_dict, self._dynamic_log_list = {}, []

        self._verb_dict = {'DEBUG': 3, 'DYNAMIC': 2, 'INFO': 1, 'ERROR': 0}
        self._is_print = lambda v: self.cf.verbose >= (v if isinstance(v, int) else self._verb_dict[v])
        self.log = lambda s, verbose=1: print(s) if self._is_print(verbose) else None
        self.ray_log = hasattr(self, 'ray_tune')

    def dict_log(self, log_dict, log_verbose='INFO'):
        self.log(' | '.join([f'{k} {v}' for k, v in log_dict.items()]), log_verbose)
        if self.cf.verbose >= self._verb_dict['DYNAMIC']:
            self._dynamic_log_list.append(log_dict)

    def log_fig(self, fig_name, fig_file):
        if self.cf.wandb_on:
            wandb.log({fig_name: wandb.Image(fig_file)})
        else:
            self.log('Figure not logged to Wandb since Wandb is off.')

    def static_log(self, log_dict):
        self._static_log_dict.update(log_dict)

    def dynamic_log(self, log_dict, log_verbose='DYNAMIC', wandb_dict=None):

        if self.cf.wandb_on:
            wandb.log(log_dict if wandb_dict is None else wandb_dict)
        # if self.ray_log:
        #     tune.report(**log_dict)
        round4 = lambda x: f'{x:.4f}'
        log_map_funcs = {'Epoch': lambda x: f'{x:03d}', 'Time':
            uf.time2str, 'Loss': round4, 'TrainAcc': round4, 'ValAcc': round4}
        log_dict.update({k: log_map_funcs[k](v) for k, v in log_dict.items() if k in log_map_funcs})
        log_dict = {k: v for k, v in log_dict.items() if k[0] != '_'}
        self.dict_log(log_dict, log_verbose)
        if self.cf.verbose >= self._verb_dict['DYNAMIC']:
            self._dynamic_log_list.append(log_dict)

    def save(self):
        out_dict = {'Static logs': self._static_log_dict, 'Dynamic logs': self._dynamic_log_list}
        json_save(out_dict, self.cf.res_file)

    def restore_prev_trial(self, mode='wandb') -> bool:
        report_func = {'wandb': wandb.log}[mode]
        if os.path.exists(self.cf.res_file):
            data = uf.json_load(self.cf.res_file)
            for log_dict in data['Dynamic logs']:
                report_func(log_dict)
            return True
        else:
            return False
