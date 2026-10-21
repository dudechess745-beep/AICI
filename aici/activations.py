# Activation functions for the LF-Transformer.
import typing as ty
import torch
import torch.nn.functional as F
from torch import Tensor

def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return reglu if name == 'reglu' else geglu if name == 'geglu' else torch.sigmoid if name == 'sigmoid' else getattr(F, name)

def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return F.relu if name == 'reglu' else F.gelu if name == 'geglu' else get_activation_fn(name)
