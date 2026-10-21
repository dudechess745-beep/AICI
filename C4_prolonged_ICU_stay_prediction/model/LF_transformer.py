# Define the LF-Transformer with column attention, row attention, and feature fusion.
import math
import typing as ty
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from aici import activations as lib
import os
from aici.knn_attention import kNNAttention

class Factorizer_col(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(self, d_numerical: int, categories: ty.Optional[ty.List[int]], d_token: int, bias: bool, f: int) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'self.category_embeddings.weight.shape={self.category_embeddings.weight.shape!r}')
        self.f = f
        self.weight = nn.Parameter(Tensor(d_numerical + f, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (0 if self.category_offsets is None else len(self.category_offsets))

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat([torch.ones(len(x_some), self.f, device=x_some.device)] + ([] if x_num is None else [x_num]), dim=1)
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat([x, self.category_embeddings(x_cat + self.category_offsets[None])], dim=1)
        if self.bias is not None:
            bias = torch.cat([torch.zeros(self.f, self.bias.shape[1], device=x.device), self.bias])
            x = x + bias[None]
        return x

class Factorizer_row(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(self, d_numerical: int, categories: ty.Optional[ty.List[int]], d_token: int, bias: bool, f: int) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            self.weight_ = nn.Parameter(Tensor(f, d_token))
            nn_init.kaiming_uniform_(self.weight_, a=math.sqrt(5))
            self.d = d_token
            print(f'self.category_embeddings.weight.shape={self.category_embeddings.weight.shape!r}')
        self.weight = nn.Parameter(Tensor(d_numerical + f, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        self.f = f

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (0 if self.category_offsets is None else len(self.category_offsets))

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat([torch.ones(len(x_num), self.f, device=x_some.device)] + ([] if x_num is None else [x_num]), dim=1)
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x_cat_ = torch.cat([torch.ones((len(x_cat), self.f, self.d), device=x_some.device) * self.weight_] + [self.category_embeddings(x_cat.permute(1, 0) + self.category_offsets[None]).permute(1, 0, 2)], dim=1)
            x = torch.cat([x, x_cat_], dim=0)
        if self.bias is not None:
            bias = torch.cat([torch.zeros(self.f, self.bias.shape[1], device=x.device), self.bias])
            x = x + bias[None]
        return x

class Tokenizer_fi(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(self, d_numerical: int, categories: ty.Optional[ty.List[int]], d_token: int, bias: bool) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'self.category_embeddings.weight.shape={self.category_embeddings.weight.shape!r}')
        self.weight = nn.Parameter(Tensor(1, d_token))
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (0 if self.category_offsets is None else len(self.category_offsets))

    def forward(self, x_num: Tensor) -> Tensor:
        x_some = x_num
        assert x_some is not None
        x_num = torch.cat([torch.ones(len(x_some), 1, x_some.shape[2], device=x_some.device)] + ([] if x_num is None else [x_num]), dim=1)
        x_num[:, :1, :] = self.weight[None] * x_num[:, :1, :].clone()
        x = x_num
        return x

class MultiheadAttention(nn.Module):

    def __init__(self, d: int, n_heads: int, dropout: float, initialization: str) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']
        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None
        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return x.reshape(batch_size, n_tokens, self.n_heads, d_head).transpose(1, 2).reshape(batch_size * self.n_heads, n_tokens, d_head)

    def forward(self, x_q: Tensor, x_kv: Tensor, key_compression: ty.Optional[nn.Linear], value_compression: ty.Optional[nn.Linear], atype: str) -> Tensor:
        x_q = x_q.to(torch.float32)
        x_kv = x_kv.to(torch.float32)
        q, k, v = (self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv))
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None
        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]
        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value).transpose(1, 2).reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        if self.W_out is not None:
            x = self.W_out(x)
        return x

class Transformer(nn.Module):

    def __init__(self, *, d_numerical: int, categories: ty.Optional[ty.List[int]], bs: int, token_bias: bool, n_layers: int, d_token: int, n_heads: int, d_ffn_factor: float, attention_dropout: float, ffn_dropout: float, residual_dropout: float, activation: str, prenormalization: bool, initialization: str, kv_compression: ty.Optional[float], kv_compression_sharing: ty.Optional[str], d_out: int, f: int, **kwargs) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)
        super().__init__()
        self.factorizer_col = Factorizer_col(d_numerical, categories, d_token, token_bias, f)
        n_tokens = self.factorizer_col.n_tokens
        self.factorizer_row = Factorizer_row(bs, categories, d_token, token_bias, f)
        n_tokens_row = self.factorizer_row.n_tokens
        self.tokenizer_fi = Tokenizer_fi(d_numerical, categories, d_token, token_bias)
        n_tokens_fi = self.factorizer_row.n_tokens

        def make_kv_compression(type):
            assert kv_compression
            if type == 'col':
                compression = nn.Linear(n_tokens, int(n_tokens * kv_compression), bias=False)
            elif type == 'row':
                compression = nn.Linear(n_tokens_row, int(n_tokens_row * kv_compression), bias=False)
            else:
                compression = nn.Linear(n_tokens, int(n_tokens * kv_compression), bias=False)
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression
        self.shared_kv_compression = make_kv_compression() if kv_compression and kv_compression_sharing == 'layerwise' else None

        def make_normalization():
            return nn.LayerNorm(d_token)
        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict({'attention_col': kNNAttention(dim=d_token, seq_len=d_numerical, num_heads=n_heads, attn_drop=attention_dropout, topk=200), 'linear0_col': nn.Linear(d_token, d_hidden * (2 if activation.endswith('glu') else 1)), 'linear1_col': nn.Linear(d_hidden, d_token), 'norm1_col': make_normalization(), 'attention_row': MultiheadAttention(d_token, n_heads, attention_dropout, initialization), 'linear0_row': nn.Linear(d_token, d_hidden * (2 if activation.endswith('glu') else 1)), 'linear1_row': nn.Linear(d_hidden, d_token), 'norm1_row': make_normalization(), 'attention_fi': MultiheadAttention(d_token, n_heads, attention_dropout, initialization), 'linear0_fi': nn.Linear(d_token, d_hidden * (2 if activation.endswith('glu') else 1)), 'linear1_fi': nn.Linear(d_hidden, d_token), 'norm1_fi': make_normalization()})
            if not prenormalization or layer_idx:
                layer['norm0_col'] = make_normalization()
                layer['norm0_row'] = make_normalization()
                layer['norm0_fi'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression('col')
                layer['key_compression_row'] = make_kv_compression('row')
                layer['key_compression_fi'] = make_kv_compression('fi')
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression('col')
                    layer['value_compression_row'] = make_kv_compression('row')
                    layer['key_compression_fi'] = make_kv_compression('fi')
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)
        self.combination_layer = nn.Linear(d_numerical, d_token)
        self.activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)
        self.f = f

    def _get_kv_compressions(self, layer):
        return (self.shared_kv_compression, self.shared_kv_compression) if self.shared_kv_compression is not None else (layer['key_compression'], layer['value_compression']) if 'key_compression' in layer and 'value_compression' in layer else (layer['key_compression'], layer['key_compression']) if 'key_compression' in layer else (None, None)

    def _get_kv_compressions_row(self, layer):
        return (self.shared_kv_compression, self.shared_kv_compression) if self.shared_kv_compression is not None else (layer['key_compression_row'], layer['value_compression_row']) if 'key_compression_row' in layer and 'value_compression_row' in layer else (layer['key_compression_row'], layer['key_compression_row']) if 'key_compression_row' in layer else (None, None)

    def _get_kv_compressions_fi(self, layer):
        return (self.shared_kv_compression, self.shared_kv_compression) if self.shared_kv_compression is not None else (layer['key_compression_fi'], layer['value_compression_fi']) if 'key_compression_row' in layer and 'value_compression_fi' in layer else (layer['key_compression_fi'], layer['key_compression_fi']) if 'key_compression_fi' in layer else (None, None)

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        x_residual = x_residual.to(torch.float32)
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.factorizer_col(x_num, x_cat)
        if x_cat is not None:
            x_ = self.factorizer_row(x_num.permute(1, 0), x_cat.permute(1, 0))
        else:
            x_ = self.factorizer_row(x_num.permute(1, 0), x_cat)
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            x_residual = self._start_residual(x, layer, '0_col')
            x_residual = layer['attention_col'](x_residual)
            x_residual = x_residual[:, :self.f] if is_last_layer else x_residual
            if is_last_layer:
                x = x[:, :x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, '0_col')
            x_residual = self._start_residual(x, layer, '1_col')
            x_residual = layer['linear0_col'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1_col'](x_residual)
            x = self._end_residual(x, x_residual, layer, '1_col')
            x_residual = self._start_residual(x_, layer, '0_row')
            x_residual = layer['attention_row'](x_residual[:, :self.f] if is_last_layer else x_residual, x_residual, *self._get_kv_compressions_row(layer), 'row')
            if is_last_layer:
                x_ = x_[:, :x_residual.shape[1]]
            x_ = self._end_residual(x_, x_residual, layer, '0_row')
            x_residual = self._start_residual(x_, layer, '1_row')
            x_residual = layer['linear0_row'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1_row'](x_residual)
            x_ = self._end_residual(x_, x_residual, layer, '1_row')
        x = x.permute(2, 0, 1) @ x_.permute(2, 1, 0)
        x = x.permute(1, 2, 0)
        x = self.tokenizer_fi(x)
        x_residual = self._start_residual(x, layer, '0_fi')
        x_residual = layer['attention_fi'](x_residual[:, :1] if is_last_layer else x_residual, x_residual, *self._get_kv_compressions_fi(layer), 'fi')
        x = x[:, :x_residual.shape[1]]
        x = self._end_residual(x, x_residual, layer, '0_fi')
        x_residual = self._start_residual(x, layer, '1_fi')
        x_residual = layer['linear0_fi'](x_residual)
        x_residual = self.activation(x_residual)
        if self.ffn_dropout:
            x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
        x_residual = layer['linear1_fi'](x_residual)
        x = self._end_residual(x, x_residual, layer, '1_fi')
        assert x.shape[1] == 1
        x = x[:, 0]
        x = x.to(torch.float32)
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x
