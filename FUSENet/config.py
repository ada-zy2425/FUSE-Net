import os
import argparse
import pprint
from datetime import datetime
from pathlib import Path
from torch import optim, nn


dir_root = Path(__file__).resolve().parent.parent
word_emb_path = dir_root / 'data' / 'glove.840B.300d.txt'
sdk_dir = dir_root / 'CMU-MultimodalSDK'
datasets_root = dir_root / 'datasets'

data_dict = {
    'mosi':     datasets_root / 'MOSI',
    'mosei':    datasets_root / 'MOSEI',
    'ch-sims':  datasets_root / 'CH-SIMS',
    'ur_funny': datasets_root / 'UR_FUNNY',
    'iemocap':  datasets_root / 'IEMOCAP', 
}

optimizer_dict = {
    'RMSprop': optim.RMSprop,
    'Adam':    optim.Adam,
    'AdamW':   optim.AdamW,
}
activation_dict = {
    'elu':        nn.ELU,
    'hardshrink': nn.Hardshrink,
    'hardtanh':   nn.Hardtanh,
    'leakyrelu':  nn.LeakyReLU,
    'prelu':      nn.PReLU,
    'relu':       nn.ReLU,
    'rrelu':      nn.RReLU,
    'tanh':       nn.Tanh,
}

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes','true','t','y','1'): return True
    if v.lower() in ('no','false','f','n','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

class Config:
    def __init__(self, **kw):
        for k, v in kw.items():
            if k == 'optimizer' and isinstance(v, str): v = optimizer_dict[v]
            if k == 'activation' and isinstance(v, str): v = activation_dict[v]
            setattr(self, k, v)

    def __str__(self):
        return "Configurations:\n" + pprint.pformat(self.__dict__)

def get_config(parse=True, **optional_kwargs):
    p = argparse.ArgumentParser()

    p.add_argument('--data',        type=str, default='mosi', help='mosi, mosei, ch-sims')
    p.add_argument('--mode',        type=str, default='train')
    p.add_argument('--use_bert',    type=str2bool, default=True)
    p.add_argument('--bert_dir',    type=str, default='roberta-large')
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    p.add_argument('--name',        type=str, default=now)


    p.add_argument('--batch_size',     type=int,   default=64)
    p.add_argument('--n_epoch',        type=int,   default=100)
    p.add_argument('--patience',       type=int,   default=10)
    p.add_argument('--optimizer',      type=str,   default='AdamW', help='Adam, AdamW, RMSprop')
    p.add_argument('--learning_rate',  type=float, default=3e-5)
    p.add_argument('--bert_learning_rate', type=float, default=1.5e-5)
    p.add_argument('--weight_decay',   type=float, default=0.01)
    p.add_argument('--warmup_steps',   type=int,   default=100)
    p.add_argument('--clip',           type=float, default=1.0)
    p.add_argument('--scheduler_type', type=str, default='plateau', help='plateau, cosine')
    p.add_argument('--T_0',            type=int, default=10, help='T_0 for CosineAnnealingWarmRestarts')
    p.add_argument('--T_mult',         type=int, default=1, help='T_mult for CosineAnnealingWarmRestarts')


    p.add_argument('--task_weight',      type=float, default=1.0)
    p.add_argument('--info_weight',      type=float, default=0.1, help="Weight for InfoNCE loss (shared-private alignment)")
    p.add_argument('--siamese_weight',   type=float, default=0.0, help="Weight for Siamese loss on noise")
    p.add_argument('--info_gain_weight', type=float, default=0.25, help="Weight for info gain loss")
    p.add_argument('--cycle_weight',     type=float, default=0.02, help="Weight for cycle consistency loss")
    p.add_argument('--recon_weight',     type=float, default=0.015, help="Weight for MRC reconstruction loss")
    p.add_argument('--vib_beta',         type=float, default=0.01, help="Beta for VIB KL-divergence in MRC")


    p.add_argument('--rnncell',         type=str,   default='gru', help="lstm or gru")
    p.add_argument('--embedding_size',  type=int,   default=300)
    p.add_argument('--hidden_size',     type=int,   default=128)
    p.add_argument('--dropout',         type=float, default=0.4)
    p.add_argument('--activation',      type=str,   default='relu')
    p.add_argument('--temperature',     type=float, default=0.3, help="Temperature for InfoNCE")
    p.add_argument('--epsilon',         type=float, default=0.15, help="Epsilon for Siamese noise loss")

    args = p.parse_args() if parse else p.parse_known_args()[0]
    ds = args.data.lower()
    if ds not in data_dict: raise ValueError(f"Unknown dataset {args.data}")
    

    if ds == 'mosi':
        args.num_classes = 1
    elif ds == 'mosei':
        args.num_classes = 1
    elif ds == 'ch-sims':
        args.num_classes = 1

    opts = vars(args)
    opts['data_dir']      = data_dict[ds]
    opts['dataset_dir']   = data_dict[ds] 
    opts['sdk_dir']       = sdk_dir
    opts['word_emb_path'] = word_emb_path
    opts.update(optional_kwargs)

    return Config(**opts)