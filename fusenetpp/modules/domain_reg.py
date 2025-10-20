# fusenetpp/modules/domain_reg.py
import torch
import torch.nn as nn

class DomainRegularizer(nn.Module):
    def __init__(self, mode='mmd', lambda_domain=0.0, gamma=1.0):
        super().__init__()
        self.mode = mode
        self.lambda_domain = lambda_domain
        self.gamma = gamma

    def mmd_rbf(self, X, Y, gamma):
        XX = torch.cdist(X, X, p=2).pow(2)
        YY = torch.cdist(Y, Y, p=2).pow(2)
        XY = torch.cdist(X, Y, p=2).pow(2)
        Kxx = torch.exp(-XX / (2*gamma))
        Kyy = torch.exp(-YY / (2*gamma))
        Kxy = torch.exp(-XY / (2*gamma))
        return Kxx.mean() + Kyy.mean() - 2*Kxy.mean()

    def forward(self, Hh_src=None, Hh_tgt=None):
        if (Hh_src is None) or (Hh_tgt is None) or (self.lambda_domain==0.0):
            return torch.tensor(0.0, device=Hh_src.device if Hh_src is not None else 'cpu')
        if self.mode == 'mmd':
            loss = self.mmd_rbf(Hh_src, Hh_tgt, self.gamma)
        else:
            loss = torch.tensor(0.0, device=Hh_src.device)
        return self.lambda_domain * loss
