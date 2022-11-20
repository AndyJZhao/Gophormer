from .config import GophormerConfig
from .train import train_gophormer
from .n_trainer import SampleNodeTrainer
from .g_trainer import SampleGraphTrainer


def get_trainer(model, g, cf, **kwargs):
    if cf.trainer == 'node':
        return SampleNodeTrainer(model=model, g=g, cf=cf, **kwargs)
    elif cf.trainer == 'graph':
        return SampleGraphTrainer(model=model, g=g, cf=cf, **kwargs)
