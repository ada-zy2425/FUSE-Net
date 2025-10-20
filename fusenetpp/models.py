# fusenetpp/models.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoConfig

from .modules.hfm import HFM
from .modules.mrc import MRC
from .modules.mdf import MDF
from .modules.domain_reg import DomainRegularizer
from .modules.probe import InfoProbe

class FUSENetPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        rnn = nn.LSTM if config.rnncell == 'lstm' else nn.GRU

        # 文本编码
        if config.use_bert:
            bert_config = AutoConfig.from_pretrained(config.bert_dir, output_hidden_states=True)
            self.bert = AutoModel.from_pretrained(config.bert_dir, config=bert_config)
            text_dim = self.bert.config.hidden_size
        else:
            self.embed = nn.Embedding(getattr(config, 'vocab_size', 10000), config.embedding_size)
            self.trnn = rnn(config.embedding_size, config.hidden_size, bidirectional=True, batch_first=True)
            text_dim = config.hidden_size * 2

        # 视/音编码
        self.vrnn = rnn(config.visual_size,  config.hidden_size, bidirectional=True, batch_first=True)
        self.arnn = rnn(config.acoustic_size, config.hidden_size, bidirectional=True, batch_first=True)

        # 线性投影到隐空间
        act = config.activation()
        self.project_t = nn.Sequential(nn.Linear(text_dim, config.hidden_size), act, nn.LayerNorm(config.hidden_size))
        self.project_v = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size), act, nn.LayerNorm(config.hidden_size))
        self.project_a = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size), act, nn.LayerNorm(config.hidden_size))

        # 三分解 / 重构 / 融合
        self.hfm = HFM(config.hidden_size, config.activation)
        self.mrc = MRC(config.hidden_size)
        self.mdf = MDF(config.hidden_size, config.activation, config.dropout)

        # 与旧 Solver 兼容的头（G/G_inv、info_gain_head_*）
        self.G     = nn.ModuleDict({m: nn.Linear(config.hidden_size, config.hidden_size) for m in ['t','v','a']})
        self.G_inv = nn.ModuleDict({m: nn.Linear(config.hidden_size, config.hidden_size) for m in ['t','v','a']})
        self.info_gain_head_s = nn.Linear(config.hidden_size, config.num_classes)
        self.info_gain_head_h = nn.Linear(config.hidden_size, config.num_classes)
        self.info_gain_head_n = nn.Linear(config.hidden_size, config.num_classes)

        # 预测头（与旧版保持相同输出维）
        self.fuse_out = nn.Linear(config.hidden_size*1, config.num_classes)

        # ++ 可选模块（先不启用，也不影响基线）
        self.probe = InfoProbe(config.hidden_size, num_outputs=config.num_classes) if getattr(config, 'use_probe', False) else None
        self.domain_reg = DomainRegularizer(
            mode=getattr(config, 'domain_mode', 'mmd'),
            lambda_domain=getattr(config, 'lambda_domain', 0.0),
            gamma=getattr(config, 'mmd_gamma', 1.0)
        )

    def _extract_features(self, seq, lengths, rnn):
        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = rnn(packed)
        unpacked, _ = pad_packed_sequence(out, batch_first=True)
        mask = torch.arange(unpacked.size(1), device=unpacked.device) < lengths.unsqueeze(1)
        masked = unpacked * mask.unsqueeze(-1)
        return masked.sum(1) / lengths.unsqueeze(1).to(unpacked.device)

    def forward(self, sentences, visual, acoustic, lengths, bs, bt, bm):
        # 文本
        if self.config.use_bert:
            bert_out = self.bert(input_ids=bs, attention_mask=bm, token_type_ids=bt)[0]
            mask = bm.unsqueeze(-1).float()
            txt = (bert_out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            txt = self._extract_features(self.embed(sentences.long()), lengths, self.trnn)
        # 视/音
        vid = self._extract_features(visual, lengths, self.vrnn)
        aud = self._extract_features(acoustic, lengths, self.arnn)

        # 投影
        proj_t, proj_v, proj_a = self.project_t(txt), self.project_v(vid), self.project_a(aud)

        # 三分解
        shared, private, noise = self.hfm(proj_t, proj_v, proj_a)

        # MRC
        mrc_out = self.mrc(shared, private, noise, targets={'t': proj_t, 'v': proj_v, 'a': proj_a})

        # 融合 + 输出
        fused = self.mdf(shared, private, noise)
        out = self.fuse_out(fused)

        # 组装旧版需要的 representations
        representations = {
            'shared': shared,
            'private': private,
            'noise': noise,
            'mrc': mrc_out
        }
        return out, representations
