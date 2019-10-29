from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_attention import TransformerLayer, Normalize, gather_nodes, gather_edges
from .protein_features import PositionalEncodings

class LanguageRNN(nn.Module):
    def __init__(self, num_letters, hidden_dim, vocab=20, num_layers=2):
        """ Graph labeling network """
        super(LanguageRNN, self).__init__()

        # Hyperparameters
        self.hidden_dim = hidden_dim
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Decoder
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, 
            num_layers=num_layers
        )
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, S, L, mask=None):
        """ Build a representation of each position in a sequence """
        h_S = self.W_s(S)
        h_V = torch.zeros([S.size(0), S.size(1), self.hidden_dim], dtype=torch.float32)
        h_S_shift = F.pad(h_S[:,0:-1], (0,0,1,0), 'constant', 0)
        h_V, _ = self.lstm(h_S_shift)
        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs

class SequenceModel(nn.Module):
    def __init__(self, num_letters, hidden_dim, num_layers=3,
        vocab=20, top_k=30, num_positional_embeddings=16):
        """ Graph labeling network """
        super(SequenceModel, self).__init__()

        # Hyperparameters
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.positional_embeddings = PositionalEmbeddings(num_positional_embeddings)

        # Embedding layers
        self.W_e = nn.Linear(num_positional_embeddings, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            GraphAttention(hidden_dim, hidden_dim*3)
            for _ in range(2 * num_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_indices(self, S):
        k = self.top_k
        N_nodes = S.size(1)
        di = (torch.arange(k) + 1).view((1, 1, -1))
        ii = torch.arange(N_nodes).view((1, -1, 1))
        # Print 
        E_idx = ii - di
        E_idx = torch.abs(E_idx)
        E_idx = torch.clamp(E_idx, 0, S.size(1))
        E_idx = E_idx.expand(S.size(0), -1,-1)
        return E_idx

    def _autoregressive_mask(self, E_idx):
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes)
        ii = ii.view((1, -1, 1))
        mask = E_idx - ii < 0
        mask = mask.type(torch.float32)

        # mask_scatter = torch.zeros(E_idx.shape[0],E_idx.shape[1],E_idx.shape[1]).scatter(-1, E_idx, mask)
        # mask_reduce = gather_edges(mask_scatter.unsqueeze(-1), E_idx).squeeze(-1)
        # plt.imshow(mask_reduce.data.numpy()[0,:,:])
        # plt.show()
        # plt.imshow(mask_scatter.data.numpy()[0,:,:])
        # plt.show()
        return mask

    def forward(self, S, L, mask=None):
        """ Build a representation of each position in a sequence """
        # V, E, E_idx = self.features(X, L, mask)
        E_idx = self._build_indices(S)
        E = self.positional_embeddings(E_idx)
        h_E = self.W_e(E)
        h_S = self.W_s(S)
        h_V = torch.zeros([S.size(0), S.size(1), self.hidden_dim], dtype=torch.float32)

        # Decoder alternates masked self-attention
        mask_attend = self._autoregressive_mask(E_idx)
        h_S_neighbors = gather_nodes(h_S, E_idx)
        for gsa in self.decoder_layers:
            h_V = gsa(h_V, h_E, E_idx, mask_V=mask, h_E_aux=h_S_neighbors, mask_attend=mask_attend)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs