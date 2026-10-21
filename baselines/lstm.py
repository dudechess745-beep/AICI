# Train, predict, and evaluate the standalone LSTM comparator.
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path[0] = str(Path(__file__).resolve().parents[1])
from torch import nn
from aici.baseline_torch import FeatureBlockRNN, train_model, load_predict

class Model(FeatureBlockRNN):

    def __init__(self, config):
        super().__init__(config['groups'], config['hidden_size'], config['dropout'], nn.LSTM)

def train(train_data, val_data, config, folder, device='cpu'):
    return train_model(Model(config), train_data, val_data, config, folder, device)

def predict(x, config, folder, device='cpu'):
    return load_predict(Model(config), x, config, folder, device)
if __name__ == '__main__':
    from aici.baseline_runner import main
    main('lstm')
