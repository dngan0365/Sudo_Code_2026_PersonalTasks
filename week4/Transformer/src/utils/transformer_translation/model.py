from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.encoding[:, : values.size(1)])


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        source_pad_id: int,
        target_pad_id: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.source_pad_id = source_pad_id
        self.target_pad_id = target_pad_id
        self.d_model = d_model
        self.source_embedding = nn.Embedding(source_vocab_size, d_model, padding_idx=source_pad_id)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=target_pad_id)
        self.position = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(d_model, target_vocab_size)

    def forward(self, source: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        source_padding_mask = source == self.source_pad_id
        target_padding_mask = target_input == self.target_pad_id
        target_mask = self._causal_mask(target_input.size(1), target_input.device)
        source_embedded = self.position(self.source_embedding(source) * math.sqrt(self.d_model))
        target_embedded = self.position(self.target_embedding(target_input) * math.sqrt(self.d_model))
        output = self.transformer(
            source_embedded,
            target_embedded,
            tgt_mask=target_mask,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        return self.output(output)

    def encode(self, source: torch.Tensor):
        source_padding_mask = source == self.source_pad_id
        source_embedded = self.position(self.source_embedding(source) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(source_embedded, src_key_padding_mask=source_padding_mask)
        return memory, source_padding_mask

    def decode_step(self, target_input: torch.Tensor, memory: torch.Tensor, source_padding_mask: torch.Tensor):
        target_padding_mask = target_input == self.target_pad_id
        target_mask = self._causal_mask(target_input.size(1), target_input.device)
        target_embedded = self.position(self.target_embedding(target_input) * math.sqrt(self.d_model))
        output = self.transformer.decoder(
            target_embedded,
            memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        return self.output(output[:, -1])

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)
