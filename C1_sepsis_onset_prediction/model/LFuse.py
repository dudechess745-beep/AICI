# Construct an LF-Transformer model from the supplied architecture parameters.
import torch
from model.LF_transformer import Transformer

def test_lf_transformer(d_numerical=288, batch_size=10, n_layers=3, d_token=320, d_ffn_factor=4, attention_dropout=0.1, ffn_dropout=0.1, residual_dropout=0.1):
    model_args = {'categories': None, 'd_out': 2, 'n_heads': 8, 'activation': 'reglu', 'prenormalization': True, 'initialization': 'kaiming', 'kv_compression': None, 'kv_compression_sharing': None, 'token_bias': True, 'f': 1}
    model = Transformer(d_numerical=d_numerical, n_layers=n_layers, d_token=d_token, d_ffn_factor=d_ffn_factor, attention_dropout=attention_dropout, ffn_dropout=ffn_dropout, residual_dropout=residual_dropout, bs=batch_size, **model_args)
    return model
