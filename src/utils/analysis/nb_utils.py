import os
import os.path as osp
import sys

import hiplot as hip
import pandas as pd

proj_path = osp.abspath(osp.dirname(__file__)).split('src')[0]
os.chdir(proj_path)
sys.path.append(proj_path + 'src')
from utils.exp.exp_utils import *
from pprint import pformat
from functools import partial
import dictdiffer

md = import_exp_settings(proj_path.strip('/').split('/')[-1])


def dict_diff(dict1, dict2, ignored_attr=[]):
    dict1 = {k: v for k, v in dict1.items() if k not in ignored_attr}
    dict2 = {k: v for k, v in dict2.items() if k not in ignored_attr}
    for diff in list(dictdiffer.diff(dict1, dict2)):
        print(diff)


def get_reproducibility_info(git_hash, model):
    commit_url, file_tree_url, raw_file_url = get_git_url_generators(git_hash)
    print(f'The results was completed in the environment: {commit_url}')
    config_file = f'src/{PROJ_NAME}/models/{model}/config.py'
    exp_file = f'src/tune/{model}/tune_meta_data.py'
    print(f":\n"
          f"- CONFIG FILE: {file_tree_url(config_file)}\n"
          f"- EXP SETTING FILE: {file_tree_url(exp_file)}\n")


def get_searched_paras(exp_dict):
    searched_paras = []
    for k, v in exp_dict.items():
        if isinstance(v, list):
            searched_paras.append(k)
        else:
            if k == 'data_spec_configs':
                is_search = lambda d: len([_ for _ in d.values() if isinstance(_, list)]) > 0
                searched_paras += [settings for settings, d in v.items() if is_search(d)]
    return list(set(searched_paras))


def get_files_by_identifiers(identifier, datasets=md.datasets):
    f_dict = {}
    for d in datasets:
        if os.path.exists(f'{d}'):
            files = [f for f in os.listdir(f'{d}') if identifier in f]
            if len(files) > 0:
                f_dict[d] = f"{d}/{files[0]}"
                continue
            elif len(files) > 1:
                print(f'{d}: found {files} by identifier {identifier}, skipped.')
        # print(f'Result file missing for {d} with identifier {identifier}, skipped.')
    return f_dict


def load_data(plt_folder, identifier):
    def _load_data(f_name, map_funcs):
        EVAL_METRIC = 'test_acc'
        df = pd.read_excel(f_name, engine='openpyxl')
        metric_names = [cname[4:] for cname in df.columns if 'avg' in cname]
        df['test_std'] = df[EVAL_METRIC].apply(
            lambda x: float(x.split('±')[1])).astype(float)
        df[EVAL_METRIC] = df[EVAL_METRIC].apply(
            lambda x: float(x.split('±')[0])).astype(float)
        for col_name, func in map_funcs.items():
            map_dict = func(df[col_name].unique())
            df[col_name] = df[col_name].apply(lambda x: map_dict[x])
        df.insert(loc=0, column='Rank', value=df.index)
        return df

    # Load experimental results
    os.chdir(plt_folder)
    map_funcs = md.map_funcs
    f_dict = get_files_by_identifiers(identifier)
    print(f'File dict to be read: \n{pformat(f_dict)}')
    data = {d: _load_data(f, map_funcs) for d, f in f_dict.items()}
    return data


def get_hiplot_func(hy_para_cf):
    def _hiplot_func(df, hy_para_cf):
        exp = hip.Experiment.from_dataframe(df[hy_para_cf['hyper_paras']])
        exp.display_data(hip.Displays.PARALLEL_PLOT).update({'hide': ['uid'] + hy_para_cf['hiplot_parallel_hidden']})
        exp.display_data(hip.Displays.TABLE).update({'hide': ['uid', 'from_uid']})
        exp.display_data(hip.Displays.TABLE).update({'order_by': [('Rank', 'asc')]})
        exp.display()

    return partial(_hiplot_func, hy_para_cf=hy_para_cf)


if __name__ == '__main__':
    # Path settings
    plt_folder = 'exp/ArchivedResults/Jan24_FeatMapLayer/'
    EVAL_METRIC = 'test_acc'
    hyperparams = ['Rank', 'sample', 'readout', 'gt_n_layers', 'batch_size', 'proximity_encoding', 'lr', 'test_acc']
    data = load_data(plt_folder, 'V14.1')
    hiplot_func = get_hiplot_func(hyperparams)
