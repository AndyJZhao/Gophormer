import os.path
import os.path as osp
import sys

proj_path = osp.abspath(osp.dirname(__file__)).split('src')[0]
sys.path.append(proj_path + 'src')
from utils.exp.exp_utils import *

_md = import_exp_settings(proj_path.strip('/').split('/')[-1])

import utils.functions as uf
import utils.exp as exp
from utils.exp.tuner import get_tuner
import argparse
import pandas as pd
import numpy as np
import ast
from functools import partial


class ExpManager:
    def __init__(self, module_name):
        self.md = exp.import_exp_settings(module_name)
        self.__dict__.update(self.md.settings)
        self.__dict__.update(self.md.__dict__)

        self.temp_res_path = lambda d: f'{self.TEMP_RES_PATH}{self.model}/{d}/'
        self.res_path = lambda d: f'{self.RES_PATH}{self.model}/{d}/'
        self.db_file = lambda d: f'{self.DB_PATH}{self.model}/{self.model}_{d}_ResData.xlsx'
        self.commit()

    def _create_column(self, df, col):
        try:
            return df['config2str'].apply(
                lambda x: ast.literal_eval(x)[col])
        except Exception as e:
            print(f'Cannot find column {col} from config2str, try to lookup from prev_default_settings')
            try:
                return getattr(self.md.para_md, col).default_val
            except KeyError:
                AssertionError(f'Parameter {col} not defined! Filled with NA')
                return np.nan

    def commit(self):
        def _load_single(res_file):
            if os.path.isfile(res_file) and res_file[-5:] == '.json':
                # Load records
                data = uf.json_load(res_file)
                if data is not None:
                    res_dict = data['Static logs']
                    sum_dict = uf.subset_dict(res_dict, self.para_group.all)
                    sum_dict['res_file'] = res_dict['res_file']  # for indexing
                    res_dict.pop('res_file', None)
                    res_dict.pop('train_command', None)
                    # FIXME, new function of dropna, by lookup config2str
                    # Check what excel looks like
                    sum_dict['config2str'] = res_dict
                    return sum_dict
            else:
                pass
                # raise ValueError(f'Illegal file name <{res_file}>')

        print('Committing temp results...')
        datasets = uf.list_dir(dir_name=f'{self.TEMP_RES_PATH}{self.model}/',
                               error_msg='No results to commit.')
        for d in datasets:
            # ! Load Temp Results
            temp_files = os.listdir(self.temp_res_path(d))
            res_list = [_load_single(f'{self.temp_res_path(d)}{_}') for _ in temp_files]
            new_res = pd.DataFrame.from_dict([_ for _ in res_list if _ is not None])

            # ! Commit Changes to Database
            resd_file = self.db_file(d)
            temp_flist = lambda df: list(df['res_file'].unique())
            if len(new_res) > 0:
                if not os.path.exists(resd_file):
                    # No previous results
                    uf.mkdir_list(resd_file)
                    new_res.to_excel(resd_file, index=False)
                    uf.remove_file(temp_flist(new_res))
                else:
                    # Merge Existing Results
                    prev_res = pd.read_excel(resd_file, engine='openpyxl')
                    merged_res = pd.concat([prev_res, new_res])
                    if len(temp_flist(merged_res)) == len(merged_res):
                        merged_res.to_excel(resd_file, index=False)
                        # ! Handle duplicates archived code
                        # uf.remove_file(temp_flist(merged_res))
                    else:  # ! Duplicate Results Found
                        # New file with old duplicates removed
                        out_df = merged_res.drop_duplicates(subset=['res_file'], keep='last')
                        out_df.to_excel(resd_file, index=False)

                        # ! Handle duplicates archived code
                        # # Old file
                        # modified_file = lambda s: resd_file.replace('.', f'_{uf.get_cur_time()}{s}.')
                        # prev_res.to_excel(modified_file('Backup'), index=False)
                        # # Conflicts
                        # conflict_df = merged_res[merged_res.duplicated(subset=['res_file'], keep=False)]
                        # conflict_df.to_excel(modified_file('Conflict'), index=False)
                        # merged_res.sort_values(['res_file'])
                        # duplicates = list(merged_res['res_file'][merged_res.duplicated(subset=["res_file"])])
                        # print(f"Found {duplicates} duplicates!")
                        #
                        # # ! Move duplicates to duplicate folder
                        # dup_path = uf.init_path(f'{self.LOG_PATH}dup_results/')
                        # for old_fname in duplicates:
                        #     os.rename(old_fname, old_fname.replace(self.temp_res_path(d), dup_path))
                        #
                        # # ! Remove other succesfully saved files
                        # uf.remove_file([f for f in new_res['res_file'] if f not in duplicates])

                print(f'Commited {len(new_res)} results to {resd_file}.')
            else:
                print(f'No new results to commit for {d}.')

    def _sort_and_save(self, df, out_prefix, metric=f'test_acc'):
        best_res = df[metric].max()
        df = df.sort_values(metric, ascending=False)
        out_file = f'{out_prefix}_{best_res}.xlsx'
        uf.mkdir_list(out_file)
        df.drop(['exp_ind', 'res_file'], axis=1).to_excel(out_file, index=False)
        return df, out_file

    def _process_raw_df(self, raw_df):
        def _na_dealing(df):
            for col in df.columns[df.isnull().any()]:
                for index in df[col][df.isnull()[col]].index:
                    try:
                        df.loc[index, col] = ast.literal_eval(df.loc[index, 'config2str'])[col]
                    except KeyError:
                        if hasattr(self.md.para_md, col):
                            df.loc[index, col] = getattr(self.md.para_md, col).default_val
                        else:
                            ValueError(f'Found previous results with attribute {col} that doesn\'t match: {index}. Please consider add default settings')
                            continue
                        # print(f'Found previous results with attribute {col} that doesn\'t match, {index} dropped.')
                        # df.drop(index)
                        # df.reset_index(inplace=True, drop=True)
            return df

        def _col_reordering(df):
            col_names = list(df.columns) + ['config2str']
            col_names.remove('config2str')
            return df[col_names]

        pg = self.para_group
        other_columns = [col for col in raw_df.columns
                         if col not in pg.metric + pg.percentage_metric + pg.annotation]
        grouped = raw_df.groupby('exp_ind')

        perc_mean_std = lambda x: f'{np.mean(x) * 100:.2f}±{np.std(x) * 100:.2f}'
        mean_std = lambda x: f'{np.mean(x):.2f}±{np.std(x):.2f}'
        df = grouped.agg({**{_: mean_std for _ in pg.metric},
                          **{_: perc_mean_std for _ in pg.percentage_metric},
                          **{_: lambda x: x.max() for _ in pg.annotation},
                          **{_: max for _ in other_columns}})
        df = df[raw_df.columns]
        df = _na_dealing(df)
        df = _col_reordering(df)
        return df

    @uf.time_logger
    def summarize(self):
        ind_func = lambda x: x.split('seed')[0].split(d)[1][1:] + x.split('seed')[1][1:]
        for d in self.datasets:
            if os.path.exists(self.db_file(d)):
                # ! Load all results
                df = pd.read_excel(self.db_file(d), engine='openpyxl')
                df['seed'] = self._create_column(df, 'seed').astype(int)
                df['exp_ind'] = df['res_file'].apply(ind_func)
                df['train_ratio'] = self._create_column(df, 'train_percentage').astype(int)

                for tr in df['train_ratio'].unique():
                    prefix = f'{self.res_path(d)}{self.model}_{d}<L{tr:02d}>'
                    raw_sub_df = df.query(f"train_ratio=={tr}", engine='python')
                    raw_sub_df = raw_sub_df.drop('train_ratio', axis=1)
                    raw_sub_df = raw_sub_df.sort_values(['exp_ind', 'seed'])
                    # ! Summarize all results
                    all_res_prefix = f'{prefix}AllRes'
                    sub_df = self._process_raw_df(raw_sub_df)
                    sub_df, res_file = self._sort_and_save(sub_df, all_res_prefix)
                    print(f'Summary of {res_file} finished.')

                    # ! Summarize by experiments
                    for exp_name in self.e_cf.exp_dict:
                        # FIXME run_times =1 ?
                        _, exp_inds = get_tuner(self.model_settings, d, exp_name, 1).check_running_status(ind_func)
                        if len(exp_inds) > 0:
                            exp_sub_df = sub_df.loc[exp_inds]
                            exp_prefix = f'{prefix}<{exp_name}>'
                            exp_df, res_file = self._sort_and_save(exp_sub_df, exp_prefix)
                            print(f'Summary of {res_file} finished.')

    def modify_exp_db(self, modify_config):
        def _rename_attribute(df, old_attr, new_attr):
            df['config2str'] = df['config2str'].apply(lambda x: x.replace(f"\'{old_attr}\'", f"\'{new_attr}\'"))
            df[new_attr] = self._create_column(df, new_attr)
            return df

        def _val_map(df, old_attr, new_attr, map_str):
            map_list = [_map_tuple.split('>') for _map_tuple in map_str.split(',')]
            for (old_val, new_val) in map_list:
                df['config2str'] = df['config2str'].apply(lambda x: x.replace(f"\'{old_attr}\': {old_val}", f"\'{new_attr}\': {new_val}"))
            return _rename_attribute(df, old_attr, new_attr)

        mode = _[0] if len((_ := modify_config.split('-'))) > 0 else 'SKIP'

        for d in self.datasets:
            if os.path.exists(db_file := self.db_file(d)):
                df = pd.read_excel(db_file, engine='openpyxl')
                # Modify exp_db
                if mode == 'SKIP':
                    pass
                elif mode == 'RenameConfig':
                    df = _rename_attribute(df, old_attr=_[1], new_attr=_[2])
                elif mode == 'NewAttrByValMap':
                    df = _val_map(df, old_attr=_[1], new_attr=_[2], map_str=_[3])
                # Reload missing attribute
                for col in self.md.para_group.all:
                    if col not in df.keys():
                        df[col] = self._create_column(df, col)
                df = df[self.md.para_group.all]
                df.to_excel(db_file, index=False)


def get_exp_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--module', type=str, default=_md.model, help='Module (model name) to summarize.')
    return parser


def summarize_exp():
    parser = get_exp_parser()
    parser.add_argument('-p', '--nb_prefix', type=str, default=_md.e_cf.exp_phase, help='nb_prefix for notebook')
    parser.add_argument('-c', '--modify_config', type=str, default='', help='how to modify')
    args = parser.parse_args()

    args.modify_config = 'RenameConfig-ft_batch_size-ft_eq_batch_size'
    args.modify_config = 'RenameConfig-prt_batch_size-prt_eq_batch_size'
    args.modify_config = 'NewAttrByValMap-ft_warmup_ratio-ft_warmup_epochs-0.1>0.5,0>0'
    args.modify_config = 'NewAttrByValMap-prt_warmup_ratio-prt_warmup_epochs-0.1>0.5,0>0'

    nb_prefix = args.nb_prefix

    em = ExpManager(args.module)
    em.modify_exp_db(args.modify_config)
    em.summarize()
    from utils.analysis import nb_analyze

    nb_analyze(args.module, f'{date}_{nb_prefix}')


if __name__ == "__main__":
    date = uf.get_cur_time().split(' ')[0].replace('-', '')
    summarize_exp()
