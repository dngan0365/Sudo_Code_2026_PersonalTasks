from __future__ import annotations

import random

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.decoder_projection = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.energy = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor, decoder_hidden: torch.Tensor, mask: torch.Tensor):
        scores = self.energy(
            torch.tanh(self.encoder_projection(encoder_outputs) + self.decoder_projection(decoder_hidden).unsqueeze(1))
        ).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float, pad_id: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, source: torch.Tensor, lengths: torch.Tensor):
        embedded = self.embedding(source)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden, cell


class AttentionSeq2Seq(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        source_pad_id: int,
        target_pad_id: int,
        target_sos_id: int,
        target_eos_id: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.target_pad_id = target_pad_id
        self.target_sos_id = target_sos_id
        self.target_eos_id = target_eos_id
        self.hidden_dim = hidden_dim
        encoder_dim = hidden_dim * 2
        self.encoder = Encoder(source_vocab_size, embedding_dim, hidden_dim, num_layers, dropout, source_pad_id)
        self.bridge_hidden = nn.Linear(encoder_dim, hidden_dim)
        self.bridge_cell = nn.Linear(encoder_dim, hidden_dim)
        self.decoder_embedding = nn.Embedding(target_vocab_size, embedding_dim, padding_idx=target_pad_id)
        self.attention = AdditiveAttention(encoder_dim=encoder_dim, decoder_dim=hidden_dim)
        self.decoder_cell = nn.LSTMCell(embedding_dim + encoder_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim + encoder_dim + embedding_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor, lengths: torch.Tensor, target: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        batch_size, target_steps = target.shape
        encoder_outputs, hidden, cell = self.encoder(source, lengths)
        decoder_hidden, decoder_cell = self._init_decoder_state(hidden, cell)
        mask = source != self.encoder.embedding.padding_idx
        logits = []
        input_token = target[:, 0]

        for step in range(1, target_steps):
            step_logits, decoder_hidden, decoder_cell, _ = self._decode_step(
                input_token,
                decoder_hidden,
                decoder_cell,
                encoder_outputs,
                mask,
            )
            logits.append(step_logits.unsqueeze(1))
            use_teacher = random.random() < teacher_forcing_ratio
            input_token = target[:, step] if use_teacher else step_logits.argmax(dim=1)
        return torch.cat(logits, dim=1)

    def generate(self, source: torch.Tensor, lengths: torch.Tensor, max_tokens: int):
        encoder_outputs, hidden, cell = self.encoder(source, lengths)
        decoder_hidden, decoder_cell = self._init_decoder_state(hidden, cell)
        mask = source != self.encoder.embedding.padding_idx
        input_token = torch.full((source.size(0),), self.target_sos_id, dtype=torch.long, device=source.device)
        generated: list[list[int]] = [[] for _ in range(source.size(0))]
        attentions = []

        for _ in range(max_tokens):
            logits, decoder_hidden, decoder_cell, weights = self._decode_step(
                input_token,
                decoder_hidden,
                decoder_cell,
                encoder_outputs,
                mask,
            )
            input_token = logits.argmax(dim=1)
            attentions.append(weights.detach().cpu())
            for row, token_id in enumerate(input_token.tolist()):
                if token_id != self.target_eos_id:
                    generated[row].append(token_id)
        return generated, attentions

    def _decode_step(self, input_token, hidden, cell, encoder_outputs, mask):
        embedded = self.dropout(self.decoder_embedding(input_token))
        context, weights = self.attention(encoder_outputs, hidden, mask)
        decoder_input = torch.cat([embedded, context], dim=1)
        hidden, cell = self.decoder_cell(decoder_input, (hidden, cell))
        logits = self.output(torch.cat([hidden, context, embedded], dim=1))
        return logits, hidden, cell, weights

    def _init_decoder_state(self, hidden, cell):
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        cell_cat = torch.cat([cell[-2], cell[-1]], dim=1)
        return torch.tanh(self.bridge_hidden(hidden_cat)), torch.tanh(self.bridge_cell(cell_cat))
