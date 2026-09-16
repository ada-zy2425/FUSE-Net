"""FUSE-Net modules corresponding to paper Sections 3.3--3.5."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


MODALITIES = ("t", "v", "a")


class FUSENet(nn.Module):
    def __init__(self, config, text_encoder=None):
        super().__init__()
        self.config = config
        if config.visual_size <= 0 or config.acoustic_size <= 0:
            raise ValueError("Feature dimensions must be established from the data before building FUSE-Net")
        if text_encoder is None:
            from transformers import AutoModel
            text_encoder = AutoModel.from_pretrained(config.bert_dir, revision=config.revision)
        self.bert = text_encoder
        text_dim = self.bert.config.hidden_size
        h = config.hidden_size
        self.vrnn = nn.GRU(config.visual_size, h, bidirectional=True, batch_first=True)
        self.arnn = nn.GRU(config.acoustic_size, h, bidirectional=True, batch_first=True)
        self.project_t = nn.Sequential(nn.Linear(text_dim, h), nn.ReLU(), nn.LayerNorm(h))
        self.project_v = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.LayerNorm(h))
        self.project_a = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.LayerNorm(h))
        self.shared = nn.Sequential(nn.Linear(h, h), nn.Sigmoid())
        for m in MODALITIES:
            setattr(self, "private_" + m, nn.Sequential(nn.Linear(h, h), nn.Sigmoid()))
            setattr(self, "noise_" + m, nn.Linear(h, h))
        self.G = nn.ModuleDict({m: nn.Linear(h, h) for m in MODALITIES})
        self.G_inv = nn.ModuleDict({m: nn.Linear(h, h) for m in MODALITIES})
        self.info_gain_head_s = nn.Linear(h, 1)
        self.info_gain_head_h = nn.Linear(h, 1)
        self.info_gain_head_n = nn.Linear(h, 1)
        self.mrc_encoders = nn.ModuleDict({m: nn.Linear(3 * h, 2 * h) for m in MODALITIES})
        self.mrc_decoders = nn.ModuleDict({m: nn.Linear(h, h) for m in MODALITIES})
        self.alpha_mlp = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.attn_mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.beta_s = nn.Parameter(torch.ones(1))
        self.beta_c = nn.Parameter(torch.ones(1))
        self.beta_n = nn.Parameter(torch.ones(1))
        self.gate_linear = nn.Linear(h, h)
        self.fuse_mlp = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(h, 1))

    @staticmethod
    def _extract_features(sequence, lengths, rnn):
        if sequence.ndim != 3 or lengths.ndim != 1 or len(lengths) != sequence.size(0):
            raise ValueError("Expected sequence [batch,time,features] and one length per sample")
        if lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Sequence lengths must be integer tensors")
        if torch.any(lengths <= 0) or torch.any(lengths > sequence.size(1)):
            raise ValueError("Sequence lengths must be positive and within their own modality")
        packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        local_lengths = lengths.to(output.device)
        mask = torch.arange(output.size(1), device=output.device)[None, :] < local_lengths[:, None]
        return (output * mask.unsqueeze(-1)).sum(1) / local_lengths[:, None]

    @staticmethod
    def sample_latent(mu, log_var, training=True):
        # A log-variance parameter requires exp(log_var / 2), not exp(log_var).
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var) if training else mu

    def forward(self, batch):
        mask = batch["attention_mask"]
        if mask.ndim != 2 or torch.any(mask.sum(1) == 0):
            raise ValueError("Every text sequence must contain a valid token")
        text_inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in batch}
        text_output = self.bert(**text_inputs)[0]
        text_mask = mask.unsqueeze(-1).to(text_output.dtype)
        text = (text_output * text_mask).sum(1) / text_mask.sum(1)
        visual = self._extract_features(batch["visual"], batch["visual_lengths"], self.vrnn)
        acoustic = self._extract_features(batch["acoustic"], batch["acoustic_lengths"], self.arnn)
        projected = {"t": self.project_t(text), "v": self.project_v(visual), "a": self.project_a(acoustic)}
        reps = {key: {} for key in ("shared", "private", "noise", "mrc", "weights", "gate")}
        for m, original in projected.items():
            shared = self.shared(original)
            private = getattr(self, "private_" + m)(original)
            noise = getattr(self, "noise_" + m)(original)
            reps["shared"][m], reps["private"][m], reps["noise"][m] = shared, private, noise
            # Equation (6): shared, specific, noise.
            mu, log_var = self.mrc_encoders[m](torch.cat((shared, private, noise), dim=1)).chunk(2, dim=1)
            z = self.sample_latent(mu, log_var, self.training)
            reps["mrc"][m] = {"target": original, "recon": self.mrc_decoders[m](z), "mu": mu, "log_var": log_var}
            # Equations (8)--(11): normalize factor branches within each modality.
            alpha = self.alpha_mlp(torch.cat((shared, private), dim=1))
            logits = torch.cat((alpha * self.beta_c * self.attn_mlp(shared),
                                alpha * self.beta_s * self.attn_mlp(private),
                                alpha * self.beta_n * self.attn_mlp(noise)), dim=1)
            reps["weights"][m] = torch.softmax(logits, dim=1)  # [batch, shared/specific/noise]
            reps["gate"][m] = torch.sigmoid(self.gate_linear(noise))
        # Equations (12)--(15), with the paper's shared/specific/noise concatenation.
        fused_shared = sum(reps["weights"][m][:, 0:1] * reps["shared"][m] for m in MODALITIES)
        fused_private = sum(reps["weights"][m][:, 1:2] * reps["private"][m] for m in MODALITIES)
        fused_noise = sum(reps["weights"][m][:, 2:3] * reps["noise"][m] * reps["gate"][m] for m in MODALITIES)
        reps["fused"] = torch.cat((fused_shared, fused_private, fused_noise), dim=1)
        return self.fuse_mlp(reps["fused"]).reshape(-1), reps
