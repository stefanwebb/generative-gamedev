"""
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International License
https://creativecommons.org/licenses/by-sa/4.0/deed.en

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from timestep_encoding import TimestepEncoding


class MotionDiffusionModel(nn.Module):
    def __init__(
        self,
        text_encoder=None,
        d_model=512,
        nhead=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation=F.gelu,
        num_layers=8,
        input_dim=263,
        max_len=5000,
    ):
        super().__init__()
        clip_dim = 512

        # Core of the transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            # batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Time (i.e. positional) embedding
        self.timestep_encoder = TimestepEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len
        )

        # Projections
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, input_dim)
        self.text_proj = nn.Linear(clip_dim, d_model)

        # Linear projection from text embedding space to Transformer space
        # TODO: How to get clip_dim automatically from text_encoder?

        # TODO: Clip
        # CLIP-version = ViT-B/32
        """
        def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
        
        TODO: Can I use HFs CLIP model loading functionality?
        """

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        bs, njoints, nfeats, nframes = x.shape
        time_emb = self.timestep_encoder(timesteps)  # [1, bs=6, d=512]

        # TODO: Masking frames, dimensions
        text_emb = self.text_proj(y["text_embed"])

        emb = text_emb + time_emb

        # TODO: Move reshaping, reordering to data pipeline (i.e. outside model code)
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        x = self.input_proj(x)

        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]

        # Don't fully understand why applying positional encoder here...
        xseq = self.timestep_encoder.pos_encoder(xseq)  # [seqlen+1, bs, d]

        output = self.encoder(xseq, src_key_padding_mask=None)[
            1:
        ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_proj(output)

        # TODO: Move reshaping, reordering to data pipeline
        output = output.reshape(nframes, bs, njoints, nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]

        return output
