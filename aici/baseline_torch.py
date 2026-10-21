# Supervised training and prediction utilities for neural comparator models.
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from .baseline_runner import finite_values
from .training import padded_forward

class FeatureBlockRNN(nn.Module):

    def __init__(self, groups, hidden, dropout, cell):
        super().__init__()
        self.groups = list(groups.values())
        self.projections = nn.ModuleList([nn.Linear(len(indices), hidden) for indices in self.groups])
        self.dropout = nn.Dropout(dropout)
        self.recurrent = cell(hidden, hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, x):
        blocks = [self.dropout(torch.relu(layer(x[:, indices]))) for indices, layer in zip(self.groups, self.projections)]
        _, state = self.recurrent(torch.stack(blocks, dim=1))
        hidden = state[0] if isinstance(state, tuple) else state
        return self.classifier(hidden[-1]).squeeze(-1)

def logits(model, x, config, lf):
    return padded_forward(model, x, config['batch_size']) if lf else model(x)

@torch.no_grad()
def probabilities(model, x, config, device, lf=False):
    model.eval()
    output = []
    values = finite_values(x, config['fill_value'])
    for start in range(0, len(values), config['batch_size']):
        z = logits(model, torch.as_tensor(values[start:start + config['batch_size']], device=device), config, lf)
        output.append((z.softmax(-1)[:, 1] if lf else z.sigmoid()).cpu().numpy())
    return np.concatenate(output) if output else np.empty(0)

def train_model(model, train, val, config, folder, device, lf=False):
    model = model.to(device)
    x = finite_values(train['X'], config['fill_value'])
    y = torch.as_tensor(train['y'], dtype=torch.long)
    counts = np.bincount(train['y'], minlength=2)
    if (counts == 0).any():
        raise ValueError('Both classes are required in the training fold.')
    if lf:
        weights = torch.tensor(len(y) / (2.0 * counts), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer_class = torch.optim.AdamW if lf else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0))
    loader = DataLoader(TensorDataset(torch.as_tensor(x), y), batch_size=config['batch_size'], shuffle=True, drop_last=False, generator=torch.Generator().manual_seed(config['seed']))
    best, stale, history = (-np.inf, 0, [])
    folder = Path(folder)
    for epoch in range(config['epochs']):
        model.train()
        total = 0.0
        for values, labels in loader:
            optimizer.zero_grad()
            z = logits(model, values.to(device), config, lf)
            loss = criterion(z, labels.to(device) if lf else labels.float().to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite baseline loss.')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        prob = probabilities(model, val['X'], config, device, lf)
        auc = float(roc_auc_score(val['y'], prob))
        history.append({'epoch': epoch, 'train_loss': total / len(y), 'validation_auc': auc})
        if auc > best:
            best, stale = (auc, 0)
            torch.save(model.state_dict(), folder / 'model.pt')
        else:
            stale += 1
        if stale >= config['patience']:
            break
    pd.DataFrame(history).to_csv(folder / 'training_history.csv', index=False)
    model.load_state_dict(torch.load(folder / 'model.pt', map_location=device, weights_only=True))
    return probabilities(model, val['X'], config, device, lf)

def load_predict(model, x, config, folder, device, lf=False):
    model = model.to(device)
    model.load_state_dict(torch.load(Path(folder) / 'model.pt', map_location=device, weights_only=True))
    return probabilities(model, x, config, device, lf)
