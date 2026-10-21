# Train, predict, and evaluate the standalone TabNet comparator.
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path[0] = str(Path(__file__).resolve().parents[1])
from pathlib import Path
import math
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from aici.baseline_runner import finite_values

def build(config, device):
    return TabNetClassifier(n_d=config['n_d'], n_a=config['n_a'], n_steps=config['n_steps'], optimizer_fn=torch.optim.Adam, optimizer_params={'lr': config['lr']}, scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params={'step_size': 50, 'gamma': 0.9}, mask_type='entmax', seed=config['seed'], device_name=device, verbose=0)

def train(train_data, val_data, config, folder, device='cpu'):
    x = finite_values(train_data['X'], config['fill_value'])
    vx = finite_values(val_data['X'], config['fill_value'])
    batch = min(config['batch_size'], len(x))
    while len(x) % batch == 1 and batch > 2:
        batch -= 1
    if batch < 2 or len(x) % batch == 1:
        raise ValueError('TabNet needs batches of at least two rows; adjust batch_size.')
    virtual = max(2, min(config['virtual_batch_size'], batch))

    def smallest_ghost_batch(size):
        width = math.ceil(size / math.ceil(size / virtual))
        return size % width or width
    sizes = [batch] + ([len(x) % batch] if len(x) % batch else [])
    while any((smallest_ghost_batch(size) < 2 for size in sizes)):
        virtual += 1
    model = build(config, device)
    model.fit(X_train=x, y_train=train_data['y'], eval_set=[(vx, val_data['y'])], eval_name=['validation'], eval_metric=['auc'], max_epochs=config['epochs'], patience=config['patience'], batch_size=batch, virtual_batch_size=virtual, weights=1, drop_last=False, num_workers=0, pin_memory=device == 'cuda')
    model.save_model(str(Path(folder) / 'model'))
    return model.predict_proba(vx)[:, 1]

def predict(x, config, folder, device='cpu'):
    model = build(config, device)
    model.load_model(str(Path(folder) / 'model.zip'))
    return model.predict_proba(finite_values(x, config['fill_value']))[:, 1]
if __name__ == '__main__':
    from aici.baseline_runner import main
    main('tabnet')
