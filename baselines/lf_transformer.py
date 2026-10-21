# Train, predict, and evaluate the standalone LF-Transformer comparator.
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path[0] = str(Path(__file__).resolve().parents[1])
from aici.training import build_model
from aici.baseline_torch import train_model, load_predict

def build(config):
    return build_model(config['cohort'], len(config['feature_names']), config)

def train(train_data, val_data, config, folder, device='cpu'):
    return train_model(build(config), train_data, val_data, config, folder, device, lf=True)

def predict(x, config, folder, device='cpu'):
    return load_predict(build(config), x, config, folder, device, lf=True)
if __name__ == '__main__':
    from aici.baseline_runner import main
    main('lf_transformer')
