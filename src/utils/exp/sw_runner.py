import os.path as osp
import sys

proj_path = osp.abspath(osp.dirname(__file__)).split('src')[0]
sys.path.append(proj_path + 'src')

from utils.exp.exp_utils import *

_md = import_exp_settings(proj_path.strip('/').split('/')[-1])

import argparse


def run_commands(args):
    init_command = f'cd {proj_path} && {load_shell_conf()["SHELL_INIT"]}'
    assert len(all_gpus := SV_INFO.gpus) % args.n_agents == 0
    n_agt = SV_INFO.n_gpus // args.n_agents
    for i in range(args.n_agents):
        gpus = ','.join(map(str, all_gpus[range(i * n_agt, (i + 1) * n_agt)]))
        screen_name = f"G{gpus}_{'_'.join(args.sweep_id.split('/')[1:])}"
        command = f'CUDA_VISIBLE_DEVICES={gpus} wandb agent {args.sweep_id}'
        # Delete previous existing screen
        os.system(f'if screen -list | grep -q {screen_name};\n then screen -S {screen_name} -X quit\nfi')
        os.system(f'screen -mdS {screen_name}\n')
        # Initialize and run
        os.system(f'screen -S "{screen_name}" -X stuff "{init_command}\r"')
        os.system(f'screen -S "{screen_name}" -X stuff "{command}\r"')
        print(f'Screen {screen_name} created, command running: {command}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_id', metavar='s', type=str, help='Sweep id to run')
    parser.add_argument('-x', '--n_agents', type=int, default=1)
    args = parser.parse_args()
    run_commands(args)
