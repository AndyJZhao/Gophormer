import os
from importlib import import_module
from utils.functions import get_src_dir
import os.path as osp
import sys
from types import SimpleNamespace
import socket
from contextlib import closing
from functools import partial
import numpy as np
import subprocess

proj_path = osp.abspath(osp.dirname(__file__)).split('src')[0]
sys.path.append(proj_path + 'src')
from tune.settings import server_settings, GIT_ACCOUNT, GIT_TOKEN

from types import SimpleNamespace
from Text2Struct.proj.settings import SV_INFO, PROJ_NAME

TUNING_FOLDER = f'tune'


class ExpConfig():
    def __init__(self, model, exp_phase, exp_ind_list, debug_dict=None):
        self.model = model
        self.exp_phase = exp_phase
        self.exp_ind_list = exp_ind_list
        debug_e_dict = {} if debug_dict is None else {f'{model}_Debug': debug_dict}
        self.exp_dict = {**debug_e_dict, **get_exp_dict(model, exp_ind_list)}

    def __str__(self):
        return f'{self.model} {self.exp_phase} experiment '


class Parameter():
    def __init__(self, default_val='NA', **kwargs):
        self.default_val = default_val
        self.para_type: str = 'NA'  # For post processing
        self.val_type: str = 'str'  # For post processing
        self.is_nb: bool = False  # For nb hiplot analysis
        self.excel_col_rank: int = 0  # 0 for normal variable;-1 for not in excel
        self.hiplot_parallel_hidden: bool = False  # Whether hidden in hiplot-parallel
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'{self.para_type} parameter with default value {self.default_val}, {f"Rank {self.excel_col_rank}" if self.excel_col_rank else "NOT reported"} in res_excel, {"" if self.is_nb else "NOT "}shown in noteboook.'


PARA_TYPE_META_DATA = {
    'numerical': {
        'para_type': 'numerical',
        'val_type': 'numerical',
        'excel_col_rank': 0,
    },
    'string': {
        'para_type': 'string',
        'val_type': 'str',
        'excel_col_rank': 0,
    },
    'metric': {
        'para_type': 'metric',
        'val_type': 'numerical',
        'excel_col_rank': 1,
        'is_nb': True,
    },
    'percentage_metric': {
        'para_type': 'percentage_metric',
        'val_type': 'numerical',
        'excel_col_rank': 2,
        'is_nb': True,
    },
    'annotation': {
        'para_type': 'annotation',
        'val_type': 'str',
        'excel_col_rank': 3,
        'is_nb': False,
    }
}


def para_generator(p_setting, *args, **kwargs):
    return Parameter(*args, **{**kwargs, **p_setting})


from functools import partial

para = SimpleNamespace(**{
    p_type: partial(Parameter, **p_setting)
    # p_type: lambda *args, **kwargs: Parameter(*args, **{**kwargs, **p_setting})
    for p_type, p_setting in PARA_TYPE_META_DATA.items()})

EXP_PARAS = {
    'training_time': para.annotation(hiplot_parallel_hidden=True),
    'git_hash': para.annotation('Not recorded', hiplot_parallel_hidden=True),
    'best_epoch': para.metric(hiplot_parallel_hidden=True),
    'val_acc': para.percentage_metric(hiplot_parallel_hidden=True),
    'test_acc': para.percentage_metric(),
    'config2str': para.annotation(),
    'res_file': para.annotation(),
    'train_command': para.annotation(hiplot_parallel_hidden=True),
}


def _get_numerical_order_map(value_list, type=float):
    values = [type(_) for _ in value_list]
    order = np.argsort(values)
    results = {value_list[id]: f'{chr(i)} {value_list[id]}' for i, id in enumerate(order)}
    return results


def init_parameter_meta_data(para_dict):
    attr_lookup = lambda attr, val: [k for k, v in para_dict.items() if getattr(v, attr) == val]
    para_group = SimpleNamespace(**{
        'all': list(para_dict.keys()),
        'notebook_paras': attr_lookup('is_nb', True),
        'hiplot_parallel_hidden': attr_lookup('hiplot_parallel_hidden', True),
        **{t: attr_lookup('para_type', t) for t in PARA_TYPE_META_DATA.keys()}
    })
    map_funcs = {
        **{_: _get_numerical_order_map for _ in para_group.numerical},
    }

    return SimpleNamespace(**para_dict), para_group, map_funcs


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def run_commands(args, commands):
    init_command = f'cd {proj_path} && {load_shell_conf()["SHELL_INIT"]}'
    for _, cmd_dict in commands.items():
        if len(cmd_dict) > 0:
            for screen_name, command in cmd_dict.items():
                # Delete previous existing screen
                os.system(f'if screen -list | grep -q {screen_name};\n then screen -S {screen_name} -X quit\nfi')
                if args.stop:
                    print(f'Screen {screen_name} deleted!')
                    continue
                else:
                    os.system(f'screen -mdS {screen_name}\n')
                    # Initialize and run
                    os.system(f'screen -S "{screen_name}" -X stuff "{init_command}\r"')
                    os.system(f'screen -S "{screen_name}" -X stuff "{command}\r"')
                    print(f'Screen {screen_name} created, command running: {command}')


# * ============================= Git Related =============================
def get_git_hash():
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip('\n')


def recover_git_full_hash(short_hash):
    return subprocess.run(['git', 'rev-parse', short_hash], stdout=subprocess.PIPE).stdout.decode('utf-8').strip('\n')


def get_git_url_generators(git_hash):
    # commit_hash = recover_git_full_hash(git_hash) # TODO delete this
    commit_hash = git_hash
    git_prefix = f'{GIT_ACCOUNT}/{PROJ_NAME}'
    commit_url = f'https://github.com/{git_prefix}/commit/{git_hash}'
    file_tree_url = lambda file_path: f'https://github.com/{git_prefix}/blob/{commit_hash}/{file_path}'

    raw_file_url = lambda file_path: f'https://raw.githubusercontent.com/{git_prefix}/{commit_hash}/{file_path}?token={GIT_TOKEN}'
    return commit_url, file_tree_url, raw_file_url


def import_exp_settings(module_name):
    return import_module(f'{TUNING_FOLDER}.{module_name}')


def get_exp_dict(module_name, exp_ind_list):
    # Put exp_dicts in different exp_files, e.g. EgoGT_V2.py, to dictionary
    tune_path = f'src/{TUNING_FOLDER}/{module_name}'
    tune_f_list = os.listdir(tune_path)
    exp_dict = {}
    for exp_ind in exp_ind_list:
        indexed_file = [tune_file.split('.py')[0] for tune_file in tune_f_list if exp_ind in tune_file]
        assert len(indexed_file) == 1, f'Invalid index {exp_ind_list}'
        exp_dict.update(import_module(f'{TUNING_FOLDER}.{module_name}.{indexed_file[0]}').exp_dict)
    return exp_dict


def load_shell_conf():
    with open(f'{get_src_dir()}utils/shell/shell_env.sh') as f:
        lines = f.readlines()
        lines = [_.strip("export ").strip('\n').split('=') for _ in lines if "export" in _]
        shell_conf = {k: v.strip('"') for k, v in lines}
    return shell_conf


gen_gpu_conf = lambda gpus_per_trail, gpus=None: SimpleNamespace(
    available_gpus=SV_INFO.gpus if gpus is None else gpus,
    gpus_per_trial=SV_INFO.n_gpus if gpus_per_trail == 'ALL' else min(SV_INFO.n_gpus, gpus_per_trail),  # All gpus
)
