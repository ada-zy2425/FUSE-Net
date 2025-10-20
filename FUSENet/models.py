import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoConfig

class FUSENet(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.config = config
        ds = config.data.lower()
        rnn = nn.LSTM if config.rnncell == 'lstm' else nn.GRU

      
        if config.use_bert:
            bert_config = AutoConfig.from_pretrained(config.bert_dir, output_hidden_states=True)
            self.bert = AutoModel.from_pretrained(config.bert_dir, config=bert_config)
            text_dim = self.bert.config.hidden_size
        else:
            self.embed = nn.Embedding(len(config.word2id), config.embedding_size)
            self.trnn = rnn(config.embedding_size, config.hidden_size, bidirectional=True, batch_first=True)
            text_dim = config.hidden_size * 2

        self.vrnn = rnn(config.visual_size, config.hidden_size, bidirectional=True, batch_first=True)
        self.arnn = rnn(config.acoustic_size, config.hidden_size, bidirectional=True, batch_first=True)

      
        self.project_t = nn.Sequential(nn.Linear(text_dim, config.hidden_size), config.activation(), nn.LayerNorm(config.hidden_size))
        self.project_v = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size), config.activation(), nn.LayerNorm(config.hidden_size))
        self.project_a = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size), config.activation(), nn.LayerNorm(config.hidden_size))

      
        self.shared    = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Sigmoid())
        self.private_t = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Sigmoid())
        self.private_v = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Sigmoid())
        self.private_a = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Sigmoid())
        self.noise_t   = nn.Linear(config.hidden_size, config.hidden_size)
        self.noise_v   = nn.Linear(config.hidden_size, config.hidden_size)
        self.noise_a   = nn.Linear(config.hidden_size, config.hidden_size)
        self.G = nn.ModuleDict({m: nn.Linear(config.hidden_size, config.hidden_size) for m in ['t','v','a']})
        self.G_inv = nn.ModuleDict({m: nn.Linear(config.hidden_size, config.hidden_size) for m in ['t','v','a']})
        self.info_gain_head_s = nn.Linear(config.hidden_size, config.num_classes)
        self.info_gain_head_h = nn.Linear(config.hidden_size, config.num_classes)
        self.info_gain_head_n = nn.Linear(config.hidden_size, config.num_classes)


        self.mrc_encoders = nn.ModuleDict({m: nn.Linear(3 * config.hidden_size, 2 * config.hidden_size) for m in ['t', 'v', 'a']})
        self.mrc_decoders = nn.ModuleDict({m: nn.Linear(config.hidden_size, config.hidden_size) for m in ['t', 'v', 'a']})


        self.alpha_mlp = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size), config.activation(), nn.Linear(config.hidden_size, 1))
        self.attn_mlp  = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), config.activation(), nn.Linear(config.hidden_size, 1))
        self.beta_s = nn.Parameter(torch.ones(1))
        self.beta_c = nn.Parameter(torch.ones(1))
        self.beta_n = nn.Parameter(torch.ones(1))
        self.gate_linear = nn.Linear(config.hidden_size, config.hidden_size)
        
      
        self.fuse_mlp = nn.Sequential(nn.Linear(3 * config.hidden_size, config.hidden_size), config.activation(), nn.Dropout(config.dropout), nn.Linear(config.hidden_size, config.num_classes))

    def _extract_features(self, seq, lengths, rnn):
        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = rnn(packed)
        unpacked, _ = pad_packed_sequence(out, batch_first=True)
        mask = torch.arange(unpacked.size(1)).to(unpacked.device) < lengths.to(unpacked.device).unsqueeze(1)
        masked_output = unpacked * mask.unsqueeze(-1)
        avg_pooled = masked_output.sum(1) / lengths.unsqueeze(1).to(unpacked.device)
        return avg_pooled

    def forward(self, sentences, visual, acoustic, lengths, bs, bt, bm):
       
        if self.config.use_bert:
            bert_out = self.bert(input_ids=bs, attention_mask=bm, token_type_ids=bt)[0]
            mask = bm.unsqueeze(-1).float()
            txt = (bert_out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            txt = self._extract_features(self.embed(sentences.long()), lengths, self.trnn)
        
        vid = self._extract_features(visual, lengths, self.vrnn)
        aud = self._extract_features(acoustic, lengths, self.arnn)

        
        proj_t, proj_v, proj_a = self.project_t(txt), self.project_v(vid), self.project_a(aud)
        
     
        private_t, private_v, private_a = self.private_t(proj_t), self.private_v(proj_v), self.private_a(proj_a)
        shared_t, shared_v, shared_a = self.shared(proj_t), self.shared(proj_v), self.shared(proj_a)
        noise_t, noise_v, noise_a = self.noise_t(proj_t), self.noise_v(proj_v), self.noise_a(proj_a)

       
        mrc_outputs = {}
        for m, s, p, n, o in zip(['t','v','a'], [shared_t, shared_v, shared_a], [private_t, private_v, private_a], [noise_t, noise_v, noise_a], [proj_t, proj_v, proj_a]):
            h_cat = torch.cat([s, p, n], dim=1)
            mu, log_var = torch.chunk(self.mrc_encoders[m](h_cat), 2, dim=1)
            z = mu + torch.randn_like(log_var.exp()) * log_var.exp()
            mrc_outputs[m] = {'target': o, 'recon': self.mrc_decoders[m](z), 'mu': mu, 'log_var': log_var}

       
        F_s, F_c, F_n = torch.zeros_like(private_t), torch.zeros_like(shared_t), torch.zeros_like(noise_t)
        for s, p, n in zip([shared_t, shared_v, shared_a], [private_t, private_v, private_a], [noise_t, noise_v, noise_a]):
            alpha = self.alpha_mlp(torch.cat([s, p], dim=1))
            gamma_s, gamma_c, gamma_n = self.attn_mlp(p), self.attn_mlp(s), self.attn_mlp(n)
            logits = torch.cat([alpha * self.beta_s * gamma_s, alpha * self.beta_c * gamma_c, alpha * self.beta_n * gamma_n], dim=1)
            w_s, w_c, w_n = torch.softmax(logits, dim=1).chunk(3, dim=1)
            F_s += w_s * p
            F_c += w_c * s
            F_n += w_n * (n * torch.sigmoid(self.gate_linear(n)))

      
        out = self.fuse_mlp(torch.cat([F_s, F_c, F_n], dim=1))
        

        representations = {
            'shared': {'t': shared_t, 'v': shared_v, 'a': shared_a},
            'private': {'t': private_t, 'v': private_v, 'a': private_a},
            'noise': {'t': noise_t, 'v': noise_v, 'a': noise_a},
            'mrc': mrc_outputs
        }
        return out, representations