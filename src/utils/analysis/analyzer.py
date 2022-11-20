import os.path as osp
import sys
import nbformat as nbf
import os

PROJ_ROOT = osp.abspath(osp.dirname(__file__)).split('src')[0]
sys.path.append(PROJ_ROOT + 'src')
from utils.functions import init_path, silent_remove
from utils.exp.exp_utils import import_exp_settings
from tune.settings import NB_ROOT, ARCHIVE_ROOT, EVAL_METRIC

md = import_exp_settings(PROJ_ROOT.strip('/').split('/')[-1])

from utils.analysis.nb_utils import get_files_by_identifiers, get_searched_paras
from utils.exp.tuner import check_exp_status
import inspect
import shutil


def create_notebook(model, plt_folder, exp_name):
    os.chdir(f'{PROJ_ROOT}{plt_folder}')
    datasets = list(get_files_by_identifiers(exp_name).keys())
    if len(datasets) == 0:
        print(f'Cannot find res_excel file for {exp_name}!! Skipped')
        return
        # Check status
    hy_para_cf = {
        'hyper_paras': ['Rank'] + get_searched_paras(md.e_cf.exp_dict[exp_name]) + md.para_group.notebook_paras + [EVAL_METRIC],
        'hiplot_parallel_hidden': md.para_group.hiplot_parallel_hidden,
    }

    nb = nbf.v4.new_notebook()
    init_cell_command = inspect.cleandoc(f"""
    # Import packages
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import ast
    import seaborn as sns
    import warnings
    import sys
    warnings.filterwarnings('ignore')
    
    # Import packages
    proj_path = '{NB_ROOT}'
    sys.path.append(proj_path + 'src')
    from utils.analysis.nb_utils import *
    
    # Load package
    hy_para_cf ={str(hy_para_cf)}
    data = load_data(plt_folder=f'{{proj_path}}{plt_folder}', identifier='{exp_name}')
    hiplot = get_hiplot_func(hy_para_cf)
    """)
    nb['cells'] = [nbf.v4.new_code_cell(init_cell_command)]

    def exp_md_cell(d):
        finished_trials, total_trials, progress = check_exp_status(md.model_settings, exp_name, d)
        md_text = inspect.cleandoc(f"""
            #{progress}\t{d:<.4s}\t{exp_name.split('_')[1]} 
            [{progress}] {exp_name} {d}: Trial status: {len(finished_trials)}/{total_trials}.
        """)
        # Print Best Config
        return nbf.v4.new_markdown_cell(md_text)

    exp_analyze_cell = lambda d: nbf.v4.new_code_cell(inspect.cleandoc(f"""hiplot(data[\'{d}\'])
print(f"Training command to recover best results:\\n{{data['{d}'].loc[0,'train_command']}}")"""))

    nb['cells'] += sum([[exp_md_cell(d), exp_analyze_cell(d)] for d in datasets], [])
    nb['cells'] += [nbf.v4.new_markdown_cell('# Reproducibility'), nbf.v4.new_code_cell(f"""get_reproducibility_info(list(data.values())[0].loc[0,'git_hash'], '{model}')""")]
    out_fname = init_path(f'{PROJ_ROOT}exp/notebooks/{exp_name}.ipynb')
    nbf.write(nb, out_fname)
    print(f'Notebook {out_fname} created.')


def nb_analyze(model, nb_prefix):
    # Pull and check files
    os.chdir(PROJ_ROOT)
    archive_root = init_path(ARCHIVE_ROOT)
    plt_folder = f'{ARCHIVE_ROOT}{nb_prefix}/'
    silent_remove(plt_folder)
    # os.system(f'cd {PROJ_ROOT} && cp -r results/{model} {archive_root}')
    # os.system(f'mv {archive_root}{model} {archive_root}{nb_prefix}')
    shutil.copytree(f'results/{model}', f'{archive_root}{nb_prefix}')

    exps = [_ for _ in md.e_cf.exp_dict.keys() if 'Debug' not in _]
    for exp_name in exps:
        create_notebook(model=model, plt_folder=plt_folder, exp_name=exp_name)


if __name__ == '__main__':
    # ! Parse arguments
    nb_prefix = 'DebugNBAutomation'
    nb_analyze(md.model, nb_prefix)
