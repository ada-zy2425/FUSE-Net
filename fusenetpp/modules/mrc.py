# fusenetpp/modules/mrc.py
import torch
import torch.nn as nn

class MRC(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.enc = nn.ModuleDict({m: nn.Linear(3*hidden_size, 2*hidden_size) for m in ['t','v','a']})
        self.dec = nn.ModuleDict({m: nn.Linear(hidden_size, hidden_size) for m in ['t','v','a']})

    def forward(self, shared, private, noise, targets):
        # targets: {'t': proj_t, 'v': proj_v, 'a': proj_a}
        out = {}
        for m in ['t','v','a']:
            h_cat = torch.cat([shared[m], private[m], noise[m]], dim=1)
            mu, log_var = torch.chunk(self.enc[m](h_cat), 2, dim=1)
            std = torch.exp(0.5 * log_var)
            z = mu + torch.randn_like(std) * std
            out[m] = {
                'target': targets[m],
                'recon': self.dec[m](z),
                'mu': mu,
                'log_var': log_var
            }
        return out
