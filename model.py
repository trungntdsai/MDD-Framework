import torch
from torch import nn
import torch.nn.functional as F
import math
from transformers import Wav2Vec2Model
from torchaudio.transforms import MelSpectrogram

class CNNBlock(nn.Module): 
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.relu(self.bn(self.conv(x))))

class BiLSTMBlock(nn.Module): 
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)  
        out = self.ln(out)     
        out = self.drop(out)
        return out

class AcousticEncoder(nn.Module): 
    def __init__(self, freq_bins=81, cnn_chs=(32, 64), lstm_hidden=256, dropout=0.2):
        super().__init__()
        self.freq_bins = freq_bins
        self.cnn1 = CNNBlock(1, cnn_chs[0], dropout=dropout)
        self.cnn2 = CNNBlock(cnn_chs[0], cnn_chs[1], dropout=dropout)

        first_lstm_input = cnn_chs[1] * freq_bins
        self.lstm1 = BiLSTMBlock(first_lstm_input, lstm_hidden, dropout=dropout)
        self.lstm2 = BiLSTMBlock(lstm_hidden * 2, lstm_hidden, dropout=dropout)
        self.lstm3 = BiLSTMBlock(lstm_hidden * 2, lstm_hidden, dropout=dropout)
        self.lstm4 = BiLSTMBlock(lstm_hidden * 2, lstm_hidden, dropout=dropout)

    def forward(self, x):
        b, t, f = x.shape
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        b, c, f, t = x.shape
        x = x.view(b, c * f, t).transpose(1, 2)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.lstm4(x)
        return x

class PhoneticEncoder(nn.Module): 
    def __init__(self, feature_bins=768, cnn_chs=(32, 64), lstm_hidden=256, dropout=0.2):
        super().__init__()
        self.feature_bins = feature_bins
        self.cnn1 = CNNBlock(1, cnn_chs[0], dropout=dropout)
        self.cnn2 = CNNBlock(cnn_chs[0], cnn_chs[1], dropout=dropout)
        first_lstm_input = cnn_chs[1] * feature_bins
        self.lstm = BiLSTMBlock(first_lstm_input, lstm_hidden, dropout=dropout)

    def forward(self, x):
        b, t, f = x.shape
        x = x.permute(0, 2, 1).unsqueeze(1)  
        x = self.cnn1(x)
        x = self.cnn2(x)
        b, c, f, t = x.shape
        x = x.view(b, c * f, t).transpose(1, 2)  
        x = self.lstm(x)
        return x

class LinguisticEncoder(nn.Module): 
    def __init__(self, vocab_size=71, embed_dim=256, lstm_hidden=256, proj_dim=1024, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden, bidirectional=True, batch_first=True)
        self.proj_k = nn.Linear(lstm_hidden * 2, proj_dim)
        self.proj_v = nn.Linear(lstm_hidden * 2, proj_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  
        o, _ = self.bilstm(x)  
        o = self.drop(o)
        hk = self.proj_k(o)
        hv = self.proj_v(o)
        return hk, hv

class AcousticPhoneticLinguistic(nn.Module):
    def __init__(self, num_classes=71, freq_bins=81, phon_feat_bins=768, lstm_hidden=256, proj_dim=1024):
        super().__init__()
        self.cal_mel = MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-100h")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        self.acoustic = AcousticEncoder(freq_bins=freq_bins, lstm_hidden=lstm_hidden)
        self.phonetic = PhoneticEncoder(feature_bins=phon_feat_bins, lstm_hidden=lstm_hidden)
        self.linguistic = LinguisticEncoder(vocab_size=num_classes, proj_dim=proj_dim, lstm_hidden=lstm_hidden)

        self.hq_dim = lstm_hidden * 4 
        self.project_hq = nn.Linear(self.hq_dim, proj_dim) if self.hq_dim != proj_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Linear(proj_dim + self.hq_dim, num_classes)

    def forward(self, wav_padded, linguistic_tokens):
        self.wav2vec2.eval() 
        with torch.no_grad():
            mels = self.cal_mel(wav_padded).permute(0, 2, 1) 
            energies = mels.sum(dim=-1, keepdim=True)
            fbanks = torch.cat([mels, energies], dim=-1)     
            
            mean = wav_padded.mean(dim=-1, keepdim=True)
            var = wav_padded.var(dim=-1, keepdim=True, unbiased=False)
            wav_norm = (wav_padded - mean) / torch.sqrt(var + 1e-7)
            
            w2v_outputs = self.wav2vec2(wav_norm)
            w2v_embs = w2v_outputs.last_hidden_state         
            
            min_time = min(fbanks.size(1), w2v_embs.size(1))
            fbanks = fbanks[:, :min_time, : ]
            w2v_embs = w2v_embs[:, :min_time, : ]
            
        Ha = self.acoustic(fbanks)  
        Hp = self.phonetic(w2v_embs)  
        
        Hq = torch.cat((Ha, Hp), dim=-1)  
        Hq_proj = self.project_hq(Hq)      
        
        HK, HV = self.linguistic(linguistic_tokens)  
        
        attn_out, attn_w = self.attn(Hq_proj, HK, HV)
        before = torch.cat((attn_out, Hq), dim=-1)
        logits = self.decoder(before)  

        return logits

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AcousticAdapterBlock(nn.Module):
    def __init__(self, hidden, dropout, kernel):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.dw   = nn.Conv1d(hidden, hidden, kernel_size=kernel,
                              padding=kernel // 2, groups=hidden)
        self.pw   = nn.Linear(hidden, hidden * 2)
        self.out  = nn.Linear(hidden * 2, hidden)
        self.norm2 = RMSNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        h = self.out(F.gelu(self.pw(h)))
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

class PromptFiLMCTC(nn.Module):


    # Config dataclass kept inline so the class is self-contained.
    from dataclasses import dataclass, field

    @dataclass
    class Config:
        pretrained_model: str        = "facebook/wav2vec2-base-100h"
        freeze_feature_extractor: bool = True
        num_heads: int               = 8
        dropout: float               = 0.1
        prompt_layers: int           = 2
        adapter_layers: int          = 4
        adapter_kernel: int          = 31

    def __init__(self, num_classes=71, cfg=None):
        """
        Args:
            num_classes : vocabulary size (same as AcousticPhoneticLinguistic's num_classes)
            cfg         : PromptFiLMCTC.Config instance; uses defaults if None
        """
        super().__init__()
        if cfg is None:
            cfg = PromptFiLMCTC.Config()
        self.cfg = cfg

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(cfg.pretrained_model)
        hidden = self.wav2vec2.config.hidden_size

        if cfg.freeze_feature_extractor:
            fe = getattr(self.wav2vec2, "feature_extractor", None)
            if fe is not None and hasattr(fe, "_freeze_parameters"):
                fe._freeze_parameters()

        heads = cfg.num_heads
        while hidden % heads != 0 and heads > 1:
            heads -= 1

        # Whether the backbone needs an explicit attention mask
        self._use_backbone_attn_mask = (
            getattr(self.wav2vec2.config, "feat_extract_norm", "group") == "layer"
        )
        self.can_emb = nn.Embedding(num_classes, hidden)
        self.can_pos = PositionalEncoding(hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prompt_encoder = nn.TransformerEncoder(enc_layer,
                                                    num_layers=cfg.prompt_layers)
        self.prompt_proj = nn.Sequential(
            RMSNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # ── FiLM-gated acoustic adapters ─────────────────────────────────────
        self.adapters = nn.ModuleList([
            AcousticAdapterBlock(hidden, cfg.dropout, cfg.adapter_kernel)
            for _ in range(cfg.adapter_layers)
        ])
        self.film  = nn.ModuleList([
            nn.Linear(hidden, hidden * 2) for _ in range(cfg.adapter_layers)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(hidden, hidden)     for _ in range(cfg.adapter_layers)
        ])

        # ── Output head ──────────────────────────────────────────────────────
        self.final_norm = RMSNorm(hidden)
        self.dropout    = nn.Dropout(cfg.dropout)
        self.head       = nn.Linear(hidden, num_classes)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _normalize_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        """Per-utterance mean/variance normalisation (same as APL pipeline)."""
        mean = wav.mean(dim=-1, keepdim=True)
        var  = wav.var(dim=-1, keepdim=True, unbiased=False)
        return (wav - mean) / torch.sqrt(var + 1e-7)

    def _wav_lengths_from_padding(self, wav_padded: torch.Tensor) -> torch.Tensor:
        """Infer true waveform lengths as the index of the last non-zero sample."""
        nonzero_mask = wav_padded.abs() > 0                    # (B, N)
        # If a row is all-zero (silence), treat the full length as valid.
        has_nonzero  = nonzero_mask.any(dim=-1)                # (B,)
        lengths = nonzero_mask.long().cumsum(dim=-1).max(dim=-1).values  # (B,)
        lengths = torch.where(has_nonzero, lengths,
                              torch.full_like(lengths, wav_padded.size(1)))
        return lengths

    def _get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Map raw-sample lengths to backbone frame lengths."""
        if hasattr(self.wav2vec2, "_get_feat_extract_output_lengths"):
            return self.wav2vec2._get_feat_extract_output_lengths(
                input_lengths
            ).long()
        lengths = input_lengths.float()
        for kernel, stride in zip(self.wav2vec2.config.conv_kernel,
                                  self.wav2vec2.config.conv_stride):
            lengths = torch.floor((lengths - kernel) / stride + 1)
        return lengths.long()

    def _encode_prompt(self, canonical_ids: torch.Tensor,
                       pad_id: int = 0) -> torch.Tensor:
        """Encode a batch of canonical sequences into a single prompt vector.

        Args:
            canonical_ids : (B, L) token IDs
            pad_id        : padding token ID (default 0)
        Returns:
            prompt : (B, hidden)
        """
        mask    = canonical_ids.eq(pad_id)          # True where padded
        all_pad = mask.all(dim=1)
        if all_pad.any():
            mask         = mask.clone()
            mask[all_pad, 0] = False                # prevent nan in softmax

        x = self.can_pos(self.can_emb(canonical_ids))
        x = self.prompt_encoder(x, src_key_padding_mask=mask)

        valid = (~mask).unsqueeze(-1).float()
        mean  = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        x_masked = x.masked_fill(mask.unsqueeze(-1), -1e4)
        maxv     = x_masked.max(dim=1).values
        maxv     = torch.where(torch.isfinite(maxv), maxv, torch.zeros_like(maxv))

        return self.prompt_proj(torch.cat([mean, maxv], dim=-1))   # (B, hidden)

    def forward(self,
                wav_padded: torch.Tensor,
                linguistic_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav_padded        : (B, N)  raw float32 waveforms, zero-padded
            linguistic_tokens : (B, L)  canonical reference token IDs

        Returns:
            logits : (B, T, num_classes)  — frame-level class scores,
                     same shape convention as AcousticPhoneticLinguistic
        """
        # 1. Normalise waveform (mirrors APL's internal normalisation)
        wav_norm = self._normalize_waveform(wav_padded)

        # 2. Build attention mask for backbone (only when required)
        backbone_mask = None
        if self._use_backbone_attn_mask:
            wav_lengths  = self._wav_lengths_from_padding(wav_padded)
            max_len      = wav_padded.size(1)
            backbone_mask = (
                torch.arange(max_len, device=wav_padded.device)
                .unsqueeze(0) < wav_lengths.unsqueeze(1)
            ).long()

        # 3. Extract acoustic features from frozen/fine-tuned backbone
        acoustic = self.wav2vec2(wav_norm,
                                 attention_mask=backbone_mask).last_hidden_state
        # acoustic : (B, T, hidden)

        # 4. Build prompt vector from canonical (linguistic) tokens
        #    linguistic_tokens plays the role of canonical_ids
        prompt = self._encode_prompt(linguistic_tokens)   # (B, hidden)

        # 5. FiLM-gated adapter stack
        x = acoustic
        for block, film_layer, gate_layer in zip(self.adapters,
                                                 self.film,
                                                 self.gates):
            h          = block(x)
            scale, shift = film_layer(prompt).unsqueeze(1).chunk(2, dim=-1)
            gate         = torch.sigmoid(gate_layer(prompt)).unsqueeze(1)
            h            = h * (1.0 + 0.5 * torch.tanh(scale)) + 0.5 * shift
            x            = x + gate * h

        # 6. Project to vocabulary — logits: (B, T, num_classes)
        logits = self.head(self.dropout(self.final_norm(x)))
        return logits