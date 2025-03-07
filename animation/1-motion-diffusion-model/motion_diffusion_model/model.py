"""
seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, # 512
                                                              nhead=self.num_heads, # 4
                                                              dim_feedforward=self.ff_size, # 1024
                                                              dropout=self.dropout, # 0.1
                                                              activation=self.activation) # gelu

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers) # 8"
                                                         ""
"""