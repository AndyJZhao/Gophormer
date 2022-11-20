import os.path as osp
import sys

proj_path = osp.abspath(osp.dirname(__file__)).split('src')[0]
sys.path.append(proj_path + 'src')

from utils.exp.exp_utils import *

_md = import_exp_settings(proj_path.strip('/').split('/')[-1])

import argparse
import numpy as np
from utils.exp.tuner import check_exp_status
from utils.functions import get_screen_list
from utils.exp.summarizer import ExpManager

gpu_x16_config = {
    'blogcatalog': list(range(16)),
    'cora': list(range(0, 8)),
    'citeseer': list(range(8, 16)),
}


def check_and_gen_cmds(args):
    """
    Each command set is composed of module_name and command key, separated by '_'.
    The module_name must match the module name in the tune folder, the command key (stored in the 'COMMAND_GENERATORS' dictionary) is the key to get the tune_commands.
    Returns: model_name and tune_commands to run
    """

    def c_and_g_single(exp_name, dataset, gpu_cf, workers, n_process, cmd_postfix):
        # ! Step 1 find trials left
        finished_trials, total_trials, progress = check_exp_status(md.model_settings, exp_name, dataset, run_times)
        if ignore_prev:
            finished_trials = []
            print('Rerunning experiments')

        trials_to_run = total_trials - len(finished_trials)
        total_workers, worker_id = [int(_) for _ in workers.split('-')]
        # ! Step 2: Generate start and end points
        if trials_to_run == 0:
            print('Experiments finished, no trial left to run.')
            return [], None
        if trials_to_run <= total_workers:
            total_workers = trials_to_run
            print(f'Only {trials_to_run} trials left to run, assigning one to each worker.')
            if total_workers < worker_id:
                print(f'Too few trials left, no need for this worker to run')
                return [], None
        trials = float(trials_to_run / total_workers)
        assert worker_id > 0, 'worker_id must be a positive integer'
        toint = lambda x: int(round(x))

        # ! Create GPU-ID sets for each screen
        def generate_single():
            n_parallel_trials = len(gpu_cf.available_gpus) // gpu_cf.gpus_per_trial
            gpus = np.random.permutation(gpu_cf.available_gpus).tolist()
            return [','.join(str(gpus[_ * gpu_cf.gpus_per_trial + __]) for __ in range(gpu_cf.gpus_per_trial)) for _ in range(n_parallel_trials)]

        gpu_list = sum([generate_single() for _ in range(n_process)], [])
        if trials < len(gpu_list):  # Only a part of gpu recourses are needed
            gpu_inds = np.random.choice(len(gpu_list), toint(trials), replace=False).tolist()
            gpu_list = [gpu_list[_] for _ in gpu_inds]
        # Permute gpu_ids, for load balance.
        print(f'{toint(trials)} {dataset} trials are required on this device.')
        start_points = np.linspace(trials * (worker_id - 1), trials * worker_id, len(gpu_list) + 1)
        end_points = start_points[1:]
        start_points = start_points[:-1]  # remove last one

        # ! Step 3: Generate tune_commands

        screen_name = lambda i: f'{exp_name}-D_{dataset}-S_{i:02d}'
        postfix = lambda i: cmd_postfix + '-v1 ' if i == 0 or args.verbose >= 0 else ' -v0 '
        cmd_dict = {screen_name(i): f'tu -g{gpu} -r{run_times} -d{dataset} -x{exp_name} -s{toint(start_points[i])} -e{toint(end_points[i])} -S{screen_name(i)} {postfix(i) } -M{args.run_mode}'
                    for i, gpu in enumerate(gpu_list)}
        return cmd_dict, progress

    exp_name, ignore_prev, workers, run_times = args.exp_name, args.ignore_prev, args.workers, args.run_times

    md = import_exp_settings(exp_name.split('_')[0])
    gpu_conf = md.e_cf.exp_dict[exp_name]['gpu_conf']

    cmd_postfix = '-i ' if args.ignore_prev else ''
    commands = {dataset: c_and_g_single(exp_name, dataset, gpu_setting, workers, args.n_process, cmd_postfix)[0]
                for dataset, gpu_setting in gpu_conf.items()}
    return commands


def check_status(args):
    """Check status of running experiments via lookup exp status of existing screens"""
    # ! Commit results first

    # ! Get running exp_names
    screen_lists = get_screen_list(module_name)
    get_exp_name = lambda m, s: f"{m}{s.split(m)[1].split('-D_')[0].split('-S_')[0]}"
    runing_exps = set([get_exp_name(module_name, s) for s in screen_lists if module_name in s])

    status = {}
    for exp in runing_exps:
        print(f'\nChecking status of {exp}')
        args.exp_name = exp
        try:
            status[exp] = check_and_gen_cmds(args)
        except Exception as e:
            import traceback
            error_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
            print(f' error message:{error_msg}')

    print('Check failed.')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', type=str, default=f'{_md.model}_Debug')
    parser.add_argument('-i', '--ignore_prev', action='store_true', help='ignore previous results or not')
    parser.add_argument('-F', '--failed_only', action="store_true", help='Recover failed results only')
    parser.add_argument('-M', '--run_mode', default='Shell', help='Debug mode: stop if run into error')
    # parser.add_argument('-f', '--force', action="store_true", help='remove previous screen')
    parser.add_argument("-v", "--verbose", default=1, type=int, help='Verbose level, higher level generates more log, -1 to shut down')
    parser.add_argument('-c', '--check_status', action="store_true", help='show running status of all running trials')
    parser.add_argument('-s', '--stop', action="store_true", help='stop the experiments')
    parser.add_argument('-n', '--n_process', type=int, default=1, help='how many process per gpu/dataset')
    parser.add_argument('-w', '--workers', type=str, default='1-1', help='total_workers and worker_id')
    parser.add_argument('-r', '--run_times', type=int, default=1, help='Run times')
    args = parser.parse_args()
    module_name = args.exp_name.split('_')[0]
    ExpManager(module_name)

    if args.check_status:
        check_status(args)
    else:
        commands = check_and_gen_cmds(args)
        run_commands(args, commands)
