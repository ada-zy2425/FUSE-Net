# fusenetpp/modules/probe.py
import torch
import torch.nn as nn

class InfoProbe(nn.Module):
    def __init__(self, dim, num_outputs=1):
        super().__init__()
        self.head = nn.Linear(dim, num_outputs)

    def forward(self, H):
        H_detach = H.detach()
        logits = self.head(H_detach)
        # 返回可记录的简单指标；MI 更复杂，后面再加
        return logits, {'probe_var': H_detach.var(dim=0).mean()}
