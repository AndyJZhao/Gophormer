import os
import sys
import logging
import numpy as np
import time
import datetime
import pytz
import socket


# * ============================= Init =============================

def exp_init(gpu_id=None, seed=0):
    """
    Functions:
    - Set GPU
    - Initialize Seeds
    - Set log level
    """
    from warnings import simplefilter
    simplefilter(action='ignore', category=DeprecationWarning)
    if gpu_id is not None:
        if gpu_id >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Torch related packages should be imported afterward setting
    init_random_state(seed)


def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    import dgl
    dgl.seed(seed)
    dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # Risk of bug


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper


def is_runing_on_local():
    try:
        host_name = socket.gethostname()
        if 'MacBook' in host_name:
            return True
    except:
        print("Unable to get Hostname and IP")
    return False


# * ============================= Print Related =============================
def subset_dict(d, sub_keys):
    return {k: d[k] for k in sub_keys if k in d}


def print_dict(d, end_string='\n\n'):
    for key in d.keys():
        if isinstance(d[key], dict):
            print('\n', end='')
            print_dict(d[key], end_string='')
        elif isinstance(d[key], int):
            print('{}: {:04d}'.format(key, d[key]), end=', ')
        elif isinstance(d[key], float):
            print('{}: {:.4f}'.format(key, d[key]), end=', ')
        else:
            print('{}: {}'.format(key, d[key]), end=', ')
    print(end_string, end='')


def block_log():
    sys.stdout = open(os.devnull, 'w')
    logger = logging.getLogger()
    logger.disabled = True


def enable_logs():
    # Restore
    sys.stdout = sys.__stdout__
    logger = logging.getLogger()
    logger.disabled = False


def print_log(log_dict):
    log_ = lambda log: f'{log:.4f}' if isinstance(log, float) else f'{log:04d}'
    print(' | '.join([f'{k} {log_(v)}' for k, v in log_dict.items()]))


def mp_list_str(mp_list):
    return '_'.join(mp_list)


# * ============================= Time Related =============================

def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%b-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime('%b-%d %H:%M:%S')


# * ============================= Itertool Related =============================

def lot_to_tol(list_of_tuple):
    # list of tuple to tuple lists
    # Note: zip(* zipped_file) is an unzip operation
    return list(map(list, zip(*list_of_tuple)))
