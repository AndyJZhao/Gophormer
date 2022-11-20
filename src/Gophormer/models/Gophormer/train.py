import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from Gophormer.functions import time_logger
from Gophormer.models.Gophormer import GophormerConfig


@time_logger
def train_gophormer(args):
    # ! Init Arguments
    cf = GophormerConfig(args).init()

    # ! Import packages
    # Note that the assignment of GPU-ID must be specified before torch/dgl is imported.
    import torch as th
    from Gophormer.proj.data import preprocess_data
    from Gophormer.models.Gophormer.model import Gophormer
    from Gophormer.models.Gophormer import get_trainer
    g, features, cf.feat_dim, cf.n_class, supervision = preprocess_data(cf.dataset, cf.train_percentage)

    # ! Train and Eval
    model = Gophormer(cf).to(cf.compute_dev)
    trainer = get_trainer(model, g, cf, features=features, sup=supervision)
    trainer.run()
    trainer.eval_and_save()
    return cf


if __name__ == "__main__":
    args = GophormerConfig().parse_args()
    train_gophormer(args)
