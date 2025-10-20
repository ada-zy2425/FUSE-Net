# fusenetpp/modules/mdf.py
import torch
import torch.nn as nn

class MDF(nn.Module):
    def __init__(self, hidden_size, activation, dropout):
        super().__init__()
        act = activation()
        self.alpha_mlp = nn.Sequential(nn.Linear(hidden_size*2, hidden_size), act, nn.Linear(hidden_size,1))
        self.attn_mlp  = nn.Sequential(nn.Linear(hidden_size, hidden_size), act, nn.Linear(hidden_size,1))
        self.beta_s = nn.Parameter(torch.ones(1))
        self.beta_c = nn.Parameter(torch.ones(1))
        self.beta_n = nn.Parameter(torch.ones(1))
        self.gate_linear = nn.Linear(hidden_size, hidden_size)
        self.fuse_head = nn.Sequential(
            nn.Linear(3*hidden_size, hidden_size), act, nn.Dropout(dropout)
        )

    def forward(self, shared, private, noise):
        # 逐模态加权求和
        mlist = ['t','v','a']
        device = next(iter(shared.values())).device
        F_s = torch.zeros_like(private['t']).to(device)
        F_c = torch.zeros_like(shared['t']).to(device)
        F_n = torch.zeros_like(noise['t']).to(device)

        for m in mlist:
            s, p, n = shared[m], private[m], noise[m]
            alpha = self.alpha_mlp(torch.cat([s, p], dim=1))
            gamma_s, gamma_c, gamma_n = self.attn_mlp(p), self.attn_mlp(s), self.attn_mlp(n)
            logits = torch.cat([alpha * self.beta_s * gamma_s,
                                alpha * self.beta_c * gamma_c,
                                alpha * self.beta_n * gamma_n], dim=1)
            w_s, w_c, w_n = torch.softmax(logits, dim=1).chunk(3, dim=1)
            F_s += w_s * p
            F_c += w_c * s
            F_n += w_n * (n * torch.sigmoid(self.gate_linear(n)))

        fused = self.fuse_head(torch.cat([F_s, F_c, F_n], dim=1))
        return fused
