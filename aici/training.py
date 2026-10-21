# Provide student losses, model construction, training, and prediction utilities.
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from .data import ROOT, TASKS, load_module
TEMPERATURE = 4.0
COSINE_THRESHOLD = 0.9

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(cohort, features, config, device='cpu'):
    module = load_module(ROOT / TASKS[cohort] / 'model/LF_transformer.py', f'model_{cohort}')
    return module.Transformer(d_numerical=features, categories=None, bs=config['batch_size'], token_bias=True, n_layers=config['n_layers'], d_token=config['d_token'], n_heads=8, d_ffn_factor=config['d_ffn_factor'], attention_dropout=config['attention_dropout'], ffn_dropout=config['ffn_dropout'], residual_dropout=config['residual_dropout'], activation='reglu', prenormalization=True, initialization='kaiming', kv_compression=None, kv_compression_sharing=None, d_out=2, f=1).to(device)

def gate(model, soft_targets_loss, label_loss):
    parameters = [p for p in model.parameters() if p.requires_grad]
    kd = torch.autograd.grad(soft_targets_loss, parameters, retain_graph=True, allow_unused=True)
    ce = torch.autograd.grad(label_loss, parameters, retain_graph=True, allow_unused=True)
    flatten = lambda gradients: torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(parameters, gradients)])
    cosine = F.cosine_similarity(flatten(kd), flatten(ce), dim=0)
    weight = 0.0 if cosine.detach().item() >= COSINE_THRESHOLD else 1.0
    return (0.5 * label_loss + 0.5 * weight * soft_targets_loss, cosine, weight)

def teacher_targets(score):
    return torch.softmax(torch.stack([-score, score], dim=-1) / TEMPERATURE, dim=-1)

def padded_forward(model, x, batch_size):
    count = len(x)
    if count > batch_size:
        raise ValueError('Batch exceeds model row embedding size')
    if count < batch_size:
        x = F.pad(x, (0, 0, 0, batch_size - count))
    return model(x, None)[:count]

def train_epoch(model, optimizer, loader, device, batch_size, class_weights):
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0
    labels_all = []
    probs_all = []
    kept = 0
    batches = 0
    for x, y, score in loader:
        x = x.to(device)
        y = y.to(device)
        score = score.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = padded_forward(model, x, batch_size)
        ce = F.cross_entropy(logits, y, weight=class_weights)
        kd = F.kl_div(F.log_softmax(logits / TEMPERATURE, dim=-1), teacher_targets(score), reduction='batchmean') * TEMPERATURE ** 2
        loss, cosine, weight = gate(model, kd, ce)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite student loss')
        loss.backward()
        optimizer.step()
        probs = logits.detach().softmax(-1)[:, 1]
        correct += int(((probs >= 0.5) == y).sum())
        count += len(y)
        total_loss += float(loss.detach()) * len(y)
        labels_all.extend(y.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())
        kept += weight
        batches += 1
    return {'loss': total_loss / count, 'accuracy': correct / count, 'auc': float(roc_auc_score(labels_all, probs_all)), 'kd_kept_fraction': kept / batches}

def student_values(x, indices):
    values = np.asarray(x[:, indices], dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError('Infinite input values are not supported.')
    return np.nan_to_num(values, nan=-7.0)

def loader_for(x, y, score, batch_size, seed):
    return DataLoader(TensorDataset(torch.as_tensor(x), torch.as_tensor(y, dtype=torch.long), torch.as_tensor(score, dtype=torch.float32)), batch_size=batch_size, shuffle=True, drop_last=False, generator=torch.Generator().manual_seed(seed))

@torch.no_grad()
def predict_model(model, x, batch_size, device):
    model.eval()
    output = []
    for start in range(0, len(x), batch_size):
        z = padded_forward(model, torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device), batch_size)
        output.append(z.softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)
