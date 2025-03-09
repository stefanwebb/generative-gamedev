"""
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International License
https://creativecommons.org/licenses/by-sa/4.0/deed.en

"""

import torch
import torch.nn as nn

# import torch.functional as F


class MotionDiffusionModel(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        num_heads=4,
        feedforward_dim=1024,
        dropout=0.1,
        activation=nn.gelu,
        num_layers=8,
        input_dim=264,  # <= TODO: Confirm dimension is correct for HumanML
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
        )

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(latent_dim, input_dim)

        # TODO: Positional encoding
        # TODO: Clip

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        # TODO: Trace through forward fn
        # bs, njoints, nfeats, nframes = x.shape
        # time_emb = self.embed_timestep(timesteps)  # [1, bs, d]
