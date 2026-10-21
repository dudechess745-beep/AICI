# Train, predict, and evaluate the standalone XGBoost comparator.
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path[0] = str(Path(__file__).resolve().parents[1])
from pathlib import Path
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from aici.data import write_json
SEARCH_SPACES = {'C1': {'max_depth': [3, 4], 'learning_rate': [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9], 'reg_alpha': [0.0, 0.005, 0.01, 0.05, 0.1], 'reg_lambda': [0.8, 1, 1.5, 2, 4]}, 'C2': {'max_depth': [3, 4], 'learning_rate': [0.002, 0.005, 0.02], 'subsample': [0.4, 0.45, 0.5, 0.55, 0.6], 'colsample_bytree': [0.4, 0.45, 0.5, 0.55, 0.6], 'reg_alpha': [0.0, 0.001, 0.01, 0.02], 'reg_lambda': [0.05, 0.1, 0.5, 1]}}

def train(train_data, val_data, config, folder, device='cpu'):
    space = SEARCH_SPACES[config['cohort']]
    folder = Path(folder)
    xt = xgb.DMatrix(train_data['X'], label=train_data['y'])
    xv = xgb.DMatrix(val_data['X'], label=val_data['y'])
    best = {'auc': -float('inf')}

    def objective(trial):
        params = {name: trial.suggest_categorical(name, values) for name, values in space.items()}
        params.update(objective='binary:logistic', eval_metric='error', tree_method='hist', device=device, seed=config['seed'], nthread=config['threads'])
        model = xgb.train(params, xt, num_boost_round=config['rounds'], evals=[(xv, 'val')], early_stopping_rounds=config['patience'], verbose_eval=False)
        probability = model.predict(xv, iteration_range=(0, model.best_iteration + 1))
        auc = float(roc_auc_score(val_data['y'], probability))
        if auc > best['auc']:
            best.update(auc=auc, params=params)
            model.save_model(folder / 'teacher.json')
        return auc
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=config['seed']))
    study.optimize(objective, n_trials=config['trials'])
    study.trials_dataframe().to_csv(folder / 'teacher_trials.csv', index=False)
    write_json(folder / 'teacher_search.json', {'best_validation_auc': best['auc'], 'params': best['params'], 'search_space': space, 'cohort': config['cohort'], 'selection_metric': 'validation_auc', 'search_space_status': 'Candidates restored from original cohort BO_TPE functions; not a historical run replay'})
    return predict(val_data['X'], config, folder, device)

def predict(x, config, folder, device='cpu'):
    model = xgb.Booster()
    model.load_model(Path(folder) / 'teacher.json')
    model.set_param({'device': device, 'nthread': config['threads']})
    return model.predict(xgb.DMatrix(x), iteration_range=(0, model.best_iteration + 1))
if __name__ == '__main__':
    from aici.baseline_runner import main
    main('xgboost')
