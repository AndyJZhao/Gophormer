from Gophormer.modules import ModelConfig, Logger
from Gophormer.proj.settings import *
from Gophormer.functions import exp_init
from types import SimpleNamespace
import os

default_cf = SimpleNamespace(**GOPHORMER_DEFAULT_SETTINGS)


class GophormerConfig(ModelConfig):

    def __init__(self, args=default_cf):
        super(GophormerConfig, self).__init__('Gophormer')
        # ! Model settings
        self.lr = 0.0005
        self.weight_decay = 1e-5
        self.dropout = 0.5
        self.in_feat_dropout = 0.0
        self.scheduler = '0_3_0.8'  # Warmup step + Patience + Decay Step

        # ! Sampler Settings
        self.n_samples = 2  # 训练时 sample/node
        self.max_samples = 30
        self.cr_weight = 0.75
        self.cr_temperature = 0.5  # Cora 0.5, Citeseer 0.3, Pubmed 0.2
        self.n_workers = 1  # >0 bugs
        # self.n_workers = 1  # 1
        self.sample = 'EgoGraph-5_1'
        self.sample_dev, self.n_workers = 'gpu', 0
        self.sample_dev, self.n_workers = 'cpu', 1
        self.batch_size = 256  # 32，16，64 比较 safe
        self.val_inference = 'FullInf'
        self.val_inference = 'SubGVote-5'
        self.val_bsz = 16
        self.global_nodes = 2
        self.trainer = 'graph'
        self.trainer = 'node'
        self.re_sample = 0

        # ! GT Settings
        self.feat_map_layer = 0
        self.gt_hidden_dim = 64  #
        self.gt_n_heads = 8
        self.gt_n_layers = 2  # 16
        self.bias = 0
        self.norm, self.scheduler = 'PreLN', 'None'
        self.norm, self.scheduler = 'PreLN', '0_3_0.8'
        self.norm, self.scheduler = 'PreLN', 'Cos10_0.2_0.5_1.2'
        self.norm = 'PreLN'
        # self.scheduler = 'CAWR_100_2'  # Cosine Anealing with Warmup Restart
        self.proximity_encoding = '0'  # None
        self.proximity_encoding = '1i'  # Max order + Identity
        self.bias_mode = 'h0_l1'  # layer specfic, head shared

        # ! Readout Settings
        self.readout = 'SPD_1_mean'
        self.readout = 'MLP_1_center'
        self.readout_dropout = 0.0

        #
        self.__dict__.update(args.__dict__)
        self.post_process_settings()

    def post_process_settings(self):
        '''
        Parse intermediate settings that shan't be saved or printed.
        '''
        tmp = SimpleNamespace()
        # Learning rate scheduler
        if self.scheduler != 'None':
            if self.scheduler.split('_')[0] == 'CAWR':
                tmp.scheduler_type = 'CAWR'
                tmp.cawr_t0, tmp.cawr_tmult = [int(_) for _ in self.scheduler[5:].split('_')]
            elif self.scheduler[:3] == 'Cos':
                # E.g. Cos10_0.2_0.5_1.2
                tmp.scheduler_type = 'Cos'
                tmp.first_cycle_epochs, tmp.warmup_factor, tmp.gamma, tmp.cycle_mult = \
                    [float(_) for _ in self.scheduler.strip('Cos').split('_')]
            else:
                tmp.scheduler_type = 'DecayAtPlateau'
                tmp.scheduler_warmup = int(self.scheduler.split('_')[0])
                tmp.scheduler_factor = float(self.scheduler.split('_')[-1])
                if len(self.scheduler.split('_')) > 2:
                    tmp.scheduler_patience = int(self.scheduler.split('_')[1]) / self.eval_freq
        tmp.stopper_patience = int(self.early_stop / self.eval_freq)
        # Sampling
        tmp.rm_self_loop = self.proximity_encoding[-1] == '-'
        tmp.sample_fn = self.sample.split('-')[0]
        tmp.fanouts = [int(_) for _ in self.sample.split('-')[1].split('_')]
        tmp.depth = len(tmp.fanouts)

        # Proximity Encoding
        tmp.K = int(self.proximity_encoding[0])
        if len(self.proximity_encoding) > 1:
            tmp.include_identity = self.proximity_encoding[1] == 'i'
        else:
            tmp.include_identity = False

        # Readout
        tmp.order_bias = tmp.K > 0 or tmp.include_identity
        # Validation inference
        tmp.val_inf_mode = self.val_inference.split('-')[0]
        tmp.val_full_inf = tmp.val_inf_mode == 'Full'
        tmp.val_n_samples = int(self.val_inference.split('-')[1]) if tmp.val_inf_mode == 'SubGVote' else 1

        self.__dict__.update(tmp.__dict__)
        self._ignored_settings += list(tmp.__dict__.keys())

        # ! Others
        self.min_eval_epochs = default_cf.min_eval_epochs[self.dataset]
        # TODO Register parameters; Instead of maintaining ignore_para_list

    def init(self):
        exp_init(self)
        self.path_init(['pe_file', 'sample_f_prefix'])

        # ! Init device
        import torch as th
        self.compute_dev = th.device("cuda:0" if self.gpus != '-1' and th.cuda.is_available() else "cpu")
        self.sample_device = th.device("cuda:0" if self.sample_dev == 'gpu' and th.cuda.is_available() else "cpu")
        self.wandb_on = False
        self.wandb_init()
        self.logger = Logger(self)
        self.logger.log(self)

        return self

    @property
    def model_cf_str(self):
        model_setting = f'lr{self.lr}-DoFeat{self.in_feat_dropout}Do{self.dropout}WD{self.weight_decay}Sch{self.scheduler}_GN{self.global_nodes}'
        sampler_setting = f'Sp{self.sample}-Inf{self.val_inference}-Bsz{self.batch_size}'
        cr_setting = f'S{self.n_samples}W{self.cr_weight}-Temp{self.cr_temperature}' if self.cr_weight > 0 else 'None'
        gt_setting = f'GT-L{self.gt_n_layers}H{self.gt_n_heads}{self.norm}-K{self.proximity_encoding}-' \
                     f'BM{self.bias_mode}_FM{int(self.feat_map_layer)}'
        readout_setting = f'RO-{self.readout}{f"-Do{self.readout_dropout}" if int(self.readout.split("_")[1]) > 0 else None}'
        return f"{model_setting}_{sampler_setting}_CR{cr_setting}_{gt_setting}_{readout_setting}"

    @property
    def parser(self):
        parser = super().parser
        parser.add_argument("-k", "--proximity_encoding", type=str, default=self.proximity_encoding,
                            help='K-hop neighbor is considered, if identity matrix is considered add "i" in the end of the string')
        parser.add_argument("--lr", type=float, default=self.lr, help='Learning rate')
        return parser

    @property
    def pe_file(self):
        return f"{MNT_TEMP_DIR}{self.model}/dgl_graph/{self.dataset}/{self.dataset}_{self.proximity_encoding}_GN-{self.global_nodes > 0}.bin"

    @property
    def sample_f_prefix(self):
        # The file to save sampled dgl_graph
        # Proximity encoding + with or without Global Nodes
        return f"{TEMP_PATH}{self.model}/sample_results/{self.dataset}/{self.dataset}_pe{self.proximity_encoding}"
