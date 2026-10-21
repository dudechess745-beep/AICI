# Train, predict, and evaluate the standalone Random Forest comparator.
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path[0] = str(Path(__file__).resolve().parents[1])
from pathlib import Path
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from aici.baseline_runner import finite_values
from aici.data import write_json

def train(train_data, val_data, config, folder, device='cpu'):
    x = finite_values(train_data['X'], config['fill_value'])
    vx = finite_values(val_data['X'], config['fill_value'])
    base = dict(config['params'])
    for key, values in config.get('params_by_fold', {}).items():
        base[key] = values[config['fold'] - 1]
    best = {'accuracy': -1.0}

    def fit_one(params):
        params = dict(params)
        if isinstance(params.get('max_features'), int):
            params['max_features'] = min(params['max_features'], x.shape[1])
        model = RandomForestClassifier(**params, n_jobs=config['threads'], random_state=config['seed'])
        model.fit(x, train_data['y'])
        score = float(accuracy_score(val_data['y'], model.predict_proba(vx)[:, 1] >= 0.5))
        if score > best['accuracy']:
            best.update(accuracy=score, params=params)
            joblib.dump(model, Path(folder) / 'model.joblib')
        return score
    search = config.get('search', {})
    if search:

        def objective(trial):
            params = dict(base)
            params.update({key: trial.suggest_categorical(key, values) for key, values in search.items()})
            return fit_one(params)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=config['seed']))
        study.optimize(objective, n_trials=config['trials'])
        study.trials_dataframe().to_csv(Path(folder) / 'search_trials.csv', index=False)
    else:
        fit_one(base)
    write_json(Path(folder) / 'selected_parameters.json', best)
    return predict(val_data['X'], config, folder, device)

def predict(x, config, folder, device='cpu'):
    model = joblib.load(Path(folder) / 'model.joblib')
    return model.predict_proba(finite_values(x, config['fill_value']))[:, 1]
if __name__ == '__main__':
    from aici.baseline_runner import main
    main('random_forest')
