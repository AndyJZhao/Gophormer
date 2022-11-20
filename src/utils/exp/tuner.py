import os.path as osp
import sys

proj_path = osp.abspath(osp.dirname(__file__)).split('src')[0]
sys.path.append(proj_path + 'src')
import multiprocessing
from utils.functions import *
from utils.exp.exp_utils import *

_md = import_exp_settings(proj_path.strip('/').split('/')[-1])

import pandas as pd
import time
from copy import deepcopy
import os
import ast
from itertools import product
from pprint import pformat
import traceback
import argparse
from types import SimpleNamespace


class Tuner():
    # Major functions
    # ✅ Maintains dataset specific tune dict
    # ✅ Tune dict to tune dataframe (para combinations)
    # ✅ Beautiful printer
    # ✅ Build-in grid search function
    # ✅ Result summarization
    # ✅ Try-catch function to deal with bugs
    # ✅ Tune report to txt.
    def __init__(self, args):
        # Init settings
        self.birth_time = get_cur_time(t_format='%m_%d-%H_%M_%S')
        self.__dict__.update(args.__dict__)
        self._md = (md := import_exp_settings(args.model))
        self.model_config, self.train_func = md.model_config, md.train_func
        self.log_on = args.verbose >= 0
        # Get tune dict
        self._td = {}
        self._path = SimpleNamespace(**md.settings)
        search_dict = md.e_cf.exp_dict[args.exp_name]

        self._iter_mode = 'iter_tune_dict'
        gpu_conf = search_dict.pop('gpu_conf', None)
        # Run in shell or inside iteration

        if 'tune_list' in search_dict:
            self.__dict__.update(search_dict['shared_cfgs'])
            # TODO More elegent way of implementation, stop using self.__dict__ and use self.common_cfg instead
            self._iter_mode = 'iter_tune_list'
            self._tune_list = search_dict['tune_list']
            if 'shared_cfgs' in search_dict:
                for d in self._tune_list:
                    d.update(search_dict['shared_cfgs'])
            print()
        else:
            if 'data_default_configs' in search_dict:
                self.update_data_specific_cand_dict(search_dict['data_default_configs'])
            self._td.update(search_dict)
            if 'data_spec_configs' in search_dict:
                self.update_data_specific_cand_dict(search_dict['data_spec_configs'])
            self._td.pop('data_spec_configs', None)
            self._td.pop('data_default_configs', None)

        # Configs
        self._searched_conf_list = list(self._td.keys())

    def update_data_specific_cand_dict(self, cand_dict):
        for k, v in cand_dict.items():
            if self.dataset in v:
                self._td.update({k: v[self.dataset]})

    # * ============================= Properties =============================

    def __str__(self):
        return f'\nExperimental config: {pformat(self.cf)}\n' \
               f'\nGrid searched parameters:{pformat(self._td)}' \
               f'\nTune_df:\n{self.tune_df}\n'

    @property
    def cf(self):
        # All configs = tune specific configs + trial configs
        return {k: v for k, v in self.__dict__.items() if k[0] != '_' and k not in ['run_mode']}

    @property
    def trial_cf(self):
        # Trial configs: configs for each trial.
        tune_global_cf = ['run_times', 'start_ind', 'reverse_iter',
                          'model', 'model_config', 'train_func', 'birth_time']
        return {k: self.cf[k] for k in self.cf if k not in tune_global_cf}

    @property
    def tune_df(self):
        # Tune dataframe: each row stands for a trial (hyper-parameter combination).
        # convert the values of parameters to list
        if self._iter_mode == 'iter_tune_list':
            tune_df = pd.DataFrame.from_records(self._tune_list)
        elif self._iter_mode == 'iter_tune_dict':
            for para in self._td:
                if not isinstance(self._td[para], list):
                    self._td[para] = [self._td[para]]
            tune_df = pd.DataFrame.from_records(dict_product(self._td))
        return tune_df

    @time_logger
    def grid_search(self, debug_mode=False):
        # ! Step 1: Report current finished trials to stdout.
        if self.ignore_prev:
            print('Ignoring previous results and rerun experiments')
            tune_df = self.tune_df
        else:
            finished_settings = self.check_running_status()[0][0]
            print(f'Found {len(finished_settings)}/{len(self.tune_df)} previous finished trials.')
            tune_df = self.tune_df.drop(finished_settings)
            if len(tune_df) == 0:
                return

        # ! Step 2: Subset trials left to run using start and end points.
        print(self)

        # Parse start and end point
        end_ind = int(self.end_point) if self.end_point > 0 else len(tune_df)
        if self.start_point < 1:
            start_ind = int(len(tune_df) * self.start_point)
        else:
            start_ind = int(self.start_point)
        end_ind = min(len(tune_df), end_ind)
        assert start_ind >= 0 and start_ind <= len(tune_df)

        tune_dict = tune_df.iloc[start_ind:end_ind].to_dict('records')
        total_trials = len(tune_dict) * self.run_times
        finished_trials, failed_settings, skipped_trials = 0, 0, 0
        outer_start_time = time.time()

        # ! Step 3: Grid search
        for i in range(len(tune_dict)):
            ind = len(tune_dict) - i - 1 if self.reverse_iter else i
            para_dict = deepcopy(self.trial_cf)
            para_dict.update(tune_dict[ind])
            print(f'\n{i}/{len(tune_dict)} <{self.exp_name}> Start tuning: {para_dict}, {get_cur_time()}')
            is_buggy_setting = False
            for seed in range(self.run_times):
                para_dict['seed'] = seed
                cf = self.model_config(SimpleNamespace(**para_dict)).init()

                if skip_results(cf.res_file, self.ignore_prev) or is_buggy_setting:
                    print(f'Skipped {cf.res_file}.')
                    total_trials -= 1
                    skipped_trials += 1
                    continue

                # ! Run trial
                if len(self.gpus) > 1 and not (debug_mode or self.run_mode == 'trainer'):
                    cmd = f'CUDA_VISIBLE_DEVICES={cf.gpus} torchrun --master_port={find_free_port()} --nproc_per_node={len(cf.gpus.split(","))} {cf.train_command.strip("python ")}'
                    print(f'RUNNING COMMAND: {cmd}')
                    os.system(command=cmd)
                else:
                    if not self.log_on: block_log()
                    if debug_mode:
                        debug_dict = {**para_dict, 'gpus': '0', 'wandb_name': f'TuDebug{self.exp_name.split("-")[-1]}'}
                        self.train_func(SimpleNamespace(**debug_dict))
                    else:
                        try:
                            self.train_func(SimpleNamespace(**para_dict))
                        except Exception as e:
                            log_file = f'log/{self.screen_name}-{self.birth_time}.log'
                            mkdir_list(log_file)
                            error_msg = ''.join(traceback.format_exception(None, e, e.__traceback__))
                            with open(log_file, 'a+') as f:
                                f.write(
                                    f'\nTrain failed at {get_cur_time()} while running {para_dict} at seed {seed},'
                                    f' error message:{error_msg}'
                                    f'Screen name: {self.screen_name}'
                                    f'Tunning command: tu -d{self.dataset} -g{self.gpus} -x{self.exp_name} -s{int(self.start_point)} -e{int(self.end_point)} -r{self.run_times} -S {self.screen_name}')
                                f.write(f'\n{"-" * 100}')
                            if not self.log_on: enable_logs()
                            print(f'Train failed, error message: {error_msg}')
                            failed_settings += 1
                            total_trials -= self.run_times
                            is_buggy_setting = True
                            continue
                if not self.log_on: enable_logs()
                finished_trials += 1
                iter_time_estimate(f'Trial finished, ', '',
                                   outer_start_time, finished_trials, total_trials)

        print(f'\n\n{"=" * 24 + " Grid Search Finished " + "=" * 24}\n'
              f'Successfully run {finished_trials} trials, skipped {skipped_trials} previous trials,'
              f'failed {failed_settings} settings.')

        if failed_settings > 0: print(f'Check {log_file} for bug reports.\n{"=" * 70}\n')
        print('!!!! TAKE UP GPU MEMORY !!!!!!')
        self.hang_up_gpu()

    def hang_up_gpu(self):
        block_log()
        # Finetune a Bert to stop GPU from being killed.
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
        import numpy as np
        from datasets import load_metric

        metric = load_metric("accuracy")
        os.environ['HF_DATASETS_DOWNLOADED_DATASETS_PATH'] = '/mnt/v-jiananzhao/hf_temp_data'
        os.environ["WANDB_DISABLED"] = "true"
        dataset = load_dataset("yelp_review_full")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        tokenized_datasets = dataset.map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True), batched=True)

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(output_dir="hf_temp", evaluation_strategy="epoch", num_train_epochs=1000, save_total_limit=1, report_to=None, per_gpu_train_batch_size=10)

        Trainer(model=model, args=training_args, train_dataset=tokenized_datasets['train'], compute_metrics=compute_metrics).train()

    # * ============================= Results Processing =============================
    def check_running_status(self, ind_func=lambda x: x):
        # Commit and check running status
        finished_trials, exp_ind_list, seed_completed = [], [], 0
        tune_df = self.tune_df
        tune_dict = tune_df.to_dict('records')
        resd_file = f'{self._path.DB_PATH}{self.model}/{self.model}_{self.dataset}_ResData.xlsx'
        if os.path.exists(resd_file):
            if len(pd.read_excel(resd_file, engine='openpyxl')) > 0:
                existing_files = pd.read_excel(resd_file, engine='openpyxl')['res_file'].unique()
                for i in range(len(tune_df)):
                    trial_finished_seed = 0
                    for seed in range(self.run_times):
                        # Check if completed.
                        para_dict = deepcopy(self.trial_cf)
                        para_dict.update(tune_dict[i])
                        para_dict['seed'] = seed
                        res_file = self.model_config(SimpleNamespace(**para_dict)).res_file

                        if res_file in existing_files:
                            trial_finished_seed += 1
                            seed_completed += 1

                    if trial_finished_seed == self.run_times:
                        finished_trials.append(i)
                        exp_ind_list.append(ind_func(res_file))
            else:  # Empty/invalid exp_db file
                remove_file(resd_file)
        return (finished_trials, len(tune_df), f'{100 * seed_completed / len(tune_df) / self.run_times:-6.2f}%'), exp_ind_list


def iter_time_estimate(prefix, postfix, start_time, iters_finished, total_iters):
    """
    Generates progress bar AFTER the ith epoch.
    Args:
        prefix: the prefix of printed string
        postfix: the postfix of printed string
        start_time: start time of the iteration
        iters_finished: finished iterations
        max_i: max iteration index
        total_iters: total iteration to run, not necessarily
            equals to max_i since some trials are skiped.

    Returns: prints the generated progress bar
    """
    cur_run_time = time.time() - start_time
    total_estimated_time = cur_run_time * total_iters / iters_finished
    print(
        f'{prefix} [{time2str(cur_run_time)}/{time2str(total_estimated_time)}, {time2str(total_estimated_time - cur_run_time)} left] {postfix} [{get_cur_time()}]')


def dict_product(d):
    keys = d.keys()
    return [dict(zip(keys, element)) for element in product(*d.values())]


def add_tune_df_common_paras(tune_df, para_dict):
    for para in para_dict:
        tune_df[para] = [para_dict[para] for _ in range(len(tune_df))]
    return tune_df


@time_logger
def run_multiple_process(func, func_arg_list):
    """
    Args:
        func: Function to run
        func_arg_list: An iterable object that contains several dict. Each dict has the input (**kwargs) of the tune_func

    Returns:

    """
    process_list = []
    for func_arg in func_arg_list:
        _ = multiprocessing.Process(target=func, kwargs=func_arg)
        process_list.append(_)
        _.start()
    for _ in process_list:
        _.join()
    return


def get_tuner(model_settings=None, dataset=None, exp_name=None, run_times=3):
    trial_dict = deepcopy(model_settings)
    trial_dict.update(
        {'dataset': dataset, 'exp_name': exp_name, 'run_times': run_times,
         'train_percentage': _md.tr_ratio_dict[dataset.split('_')[0]], 'wandb_name': 'OFF', 'verbose': 0})
    # print(trial_dict)
    return Tuner(SimpleNamespace(**trial_dict))


def check_exp_status(model_settings, exp_name, dataset, run_times=1):
    os.chdir(proj_path)
    (finished_trials, total_trials, progress), _ = \
        get_tuner(model_settings=model_settings, exp_name=exp_name, run_times=run_times, dataset=dataset).check_running_status()
    print(f'[{progress}] {exp_name} {dataset:<.4s}: Trial status: {len(finished_trials)}/{total_trials}.')
    return finished_trials, total_trials, progress


def load_dict_results(f_name):
    # Init records
    parameters = {}
    metric_set = None
    eid = 0
    with open(f_name, 'r') as f:
        res_lines = f.readlines()
        for line in res_lines:
            if line[0] == '{':
                d = ast.literal_eval(line.strip('\n'))
                if 'model' in d.keys():  # parameters
                    eid += 1
                    parameters[eid] = line.strip('\n')
                elif 'avg_' in list(d.keys())[0] or 'std_' in list(d.keys())[0]:
                    pass
                else:  # results
                    if metric_set == None:
                        metric_set = list(d.keys())
                        for m in metric_set:  # init metric dict
                            exec(f'{m.replace("-", "")}=dict()')
                    for m in metric_set:
                        exec(f'{m.replace("-", "")}[eid]=float(d[\'{m}\'])')
    metric_set_str = str(metric_set).replace('\'', '').strip('[').strip(']').replace("-", "")
    exec(f'out_list_ = [parameters,{metric_set_str}]', globals(), locals())
    out_list = locals()["out_list_"]
    out_df = pd.DataFrame.from_records(out_list).T
    out_df.columns = ['parameters', *metric_set]
    return out_df, metric_set


def skip_results(res_file, ignore_prev):
    if os.path.isfile(res_file):
        if ignore_prev:
            os.remove(res_file)
            return False
        else:
            return True


@time_logger
def tune_model(dataset=_md.datasets[0],
               run_times=3, exp_name=f'{_md.model}_Debug'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default=dataset)
    parser.add_argument('-t', '--train_percentage', type=int, default=_md.tr_ratio_dict[dataset.split('_')[0]])
    parser.add_argument('-r', '--run_times', type=int, default=run_times)
    parser.add_argument('-x', '--exp_name', type=str, default=exp_name)
    parser.add_argument('-s', '--start_point', type=float, default=0)
    parser.add_argument('-e', '--end_point', type=float, default=-1)
    parser.add_argument('-g', '--gpus', type=str, default='0')
    parser.add_argument("-v", "--verbose", default=1, type=int, help='Verbose level, higher level generates more log, -1 to shut down')
    parser.add_argument('-i', '--ignore_prev', action='store_true', help='ignore previous results or not')
    parser.add_argument('-V', '--reverse_iter', action='store_true', help='reverse iter or not')
    parser.add_argument('-D', '--debug_mode', action='store_true', help='Debug mode: stop if run into error')
    parser.add_argument('-M', '--run_mode', default='Shell', help='Debug mode: stop if run into error')
    parser.add_argument('-S', '--screen_name', type=str, default=exp_name)
    args = parser.parse_args()
    if is_runing_on_local(): args.gpu = -1

    args.verbose = 3 if args.debug_mode else args.verbose
    args.model = args.exp_name.split('_')[0]
    tuner = Tuner(args)
    tuner.grid_search(args.debug_mode)


if __name__ == '__main__':
    tune_model()
