# Resolve computing devices from configuration and command-line arguments.
import torch

def resolve_device(configured, override=None, cpu_only=False):
    device = override if override is not None else configured
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if cpu_only:
        return 'cpu'
    if device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA is configured but unavailable. Use --device cpu for CPU checks, or install CUDA-enabled PyTorch on a compatible GPU machine.')
    return device
