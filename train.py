import sys
import torch
import numpy as np
from config import get_config, Config
from data_loader import get_loader
from solver import Solver


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run():

    train_cfg = get_config(parse=True)
    print("---- Training configuration ----")
    print(train_cfg)

    base_opts = {k: v for k, v in train_cfg.__dict__.items() if k not in ('mode',)}
    dev_cfg  = Config(**{**base_opts, 'mode': 'dev'})
    test_cfg = Config(**{**base_opts, 'mode': 'test'})

    train_loader = get_loader(train_cfg, shuffle=True)
    dev_loader = get_loader(dev_cfg, shuffle=False)
    test_loader = get_loader(test_cfg, shuffle=False)


    solver = Solver(train_cfg, dev_cfg, test_cfg, train_loader, dev_loader, test_loader, is_train=True)
    solver.build()
    solver.train()

if __name__ == '__main__':
    run()