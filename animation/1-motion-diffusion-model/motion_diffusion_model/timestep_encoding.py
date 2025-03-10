"""
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International License
https://creativecommons.org/licenses/by-sa/4.0/deed.en

"""

from positional_encoding import PositionalEncoding
import torch
import torch.nn as nn


class TimestepEncoding(nn.Module):
    """
    Embedding function for denoising diffusion timestep.

    Essentially, adds a feedforward network on top of standard transformer
    positional embeddings.

    """

    def __init__(
        self, d_model=512, dropout=0.1, max_len=5000
    ):  # TODO: Default argument values
        super().__init__()

        self.pos_encoder = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len
        )

        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    # TODO: Simplify!
    def forward(self, timesteps):
        return self.time_embed(self.pos_encoder.pe[timesteps]).permute(1, 0, 2)
