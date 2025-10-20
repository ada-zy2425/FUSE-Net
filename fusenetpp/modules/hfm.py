import torch.nn as nn

class HFM(nn.Module):
    def __init__(self, hidden_size, activation):
        super().__init__()
        act = activation()
        self.shared    = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.private_t = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.private_v = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.private_a = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.noise_t   = nn.Linear(hidden_size, hidden_size)
        self.noise_v   = nn.Linear(hidden_size, hidden_size)
        self.noise_a   = nn.Linear(hidden_size, hidden_size)

    def forward(self, proj_t, proj_v, proj_a):
        shared_t = self.shared(proj_t); shared_v = self.shared(proj_v); shared_a = self.shared(proj_a)
        private_t = self.private_t(proj_t); private_v = self.private_v(proj_v); private_a = self.private_a(proj_a)
        noise_t = self.noise_t(proj_t); noise_v = self.noise_v(proj_v); noise_a = self.noise_a(proj_a)
        return (
            {'t': shared_t,  'v': shared_v,  'a': shared_a},
            {'t': private_t, 'v': private_v, 'a': private_a},
            {'t': noise_t,   'v': noise_v,   'a': noise_a}
        )
