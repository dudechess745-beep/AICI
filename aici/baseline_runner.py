# Shared training, prediction, and evaluation workflow for comparator models.
import argparse
import copy
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from .data import ROOT, load_arrays, make_folds, assert_separate, subset, read_json, write_json
from .evaluation import evaluate_csv, validate_probability
from .pipeline import youden_threshold
from .training import seed_all
from .devices import resolve_device
MODELS = ('random_forest', 'xgboost', 'tabnet', 'lstm', 'gru', 'lf_transformer')

def model_module(name):
    if name not in MODELS:
        raise ValueError(f'Unknown baseline: {name}')
    return importlib.import_module(f'baselines.{name}')

def checked_data(path):
    data = load_arrays(path)
    n, width = data['X'].shape
    if not n or not width or len(data['feature_names']) != width:
        raise ValueError('Empty data or incorrect feature names.')
    if len(set(data['feature_names'])) != width:
        raise ValueError('Duplicate feature names.')
    if any((len(data[k]) != n for k in ('y', 'hour', 'event_id'))):
        raise ValueError('Metadata length differs from input rows.')
    if np.isinf(data['X']).any():
        raise ValueError('Infinite features are not supported; fix upstream processing.')
    return data

def finite_values(x, fill):
    values = np.asarray(x, dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError('Infinite features are not supported.')
    return np.nan_to_num(values, nan=float(fill))

def feature_groups(names):
    clinical = {'ShockIndex', 'BUN/CR', 'SaO2/FiO2', 'SOFA', 'SOFA_deterioration', 'SOFA_max_24hrs', 'qSOFA', 'septic_shock', 'MEWS'}
    groups = {k: [] for k in ('raw_and_missingness', 'window', 'empirical', 'clinical')}
    for i, name in enumerate(names):
        if name in clinical:
            key = 'clinical'
        elif name.endswith('_signature'):
            key = 'empirical'
        elif name.endswith(('_max', '_min', '_mean', '_median', '_std')):
            key = 'window'
        else:
            key = 'raw_and_missingness'
        groups[key].append(i)
    if any((not indices for indices in groups.values())):
        raise ValueError('RNN inputs need all four named feature blocks. Use the prepared engineered features, or set feature_groups explicitly in the baseline config.')
    return groups

def resolve_settings(config, model, cohort, names):
    settings = copy.deepcopy(config['models'][model])
    overrides = settings.pop('cohorts', {})
    settings.update(overrides.get(cohort, {}))
    settings.update(cohort=cohort, feature_names=list(names), seed=config.get('seed', 3407), threads=config.get('threads', 4))
    for key in ('threads', 'epochs', 'batch_size', 'patience', 'trials', 'rounds', 'hidden_size', 'n_layers', 'd_token', 'n_d', 'n_a', 'n_steps'):
        if key in settings and (not isinstance(settings[key], int) or settings[key] < 1):
            raise ValueError(f'{key} must be a positive integer.')
    if model == 'lf_transformer' and settings['d_token'] % 8:
        raise ValueError('LF-Transformer d_token must be divisible by eight attention heads.')
    if model in ('lstm', 'gru'):
        configured = settings.get('feature_groups')
        if configured:
            groups = {key: [list(names).index(name) for name in values] for key, values in configured.items()}
        else:
            groups = feature_groups(names)
        indices = [i for values in groups.values() for i in values]
        if any((not v for v in groups.values())) or sorted(indices) != list(range(len(names))):
            raise ValueError('Feature groups must partition all columns exactly once.')
        settings['groups'] = groups
    return settings

def train(model, cohort, data_dir, output, config=None, device=None):
    if cohort not in ('C1', 'C2'):
        raise ValueError('These six paper comparators are defined for C1/C2.')
    cfg = read_json(config or ROOT / 'configs/baselines.json') if not isinstance(config, dict) else config
    data_dir, out = (Path(data_dir), Path(output))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty output directory to avoid mixing model runs.')
    tr = checked_data(data_dir / 'train.npz')
    if (data_dir / 'val.npz').exists():
        assert_separate(tr, checked_data(data_dir / 'val.npz'))
    pairs = make_folds(tr, cfg.get('seed', 3407))
    settings = resolve_settings(cfg, model, cohort, tr['feature_names'].tolist())
    device = resolve_device(settings.get('device', 'cpu'), device, cpu_only=model == 'random_forest')
    settings['device'] = device
    module = model_module(model)
    torch.set_num_threads(settings['threads'])
    out.mkdir(parents=True, exist_ok=True)
    run = {'model': model, 'cohort': cohort, 'settings': settings, 'folds': 5, 'feature_names': tr['feature_names'].tolist(), 'status': 'incomplete', 'decision_rule': 'C1: fold Youden thresholds and majority vote; C2: mean probability >= 0.5', 'configuration_status': cfg.get('configuration_status', 'runnable reconstruction')}
    write_json(out / 'baseline_run.json', run)
    oof = np.empty(len(tr['y']), dtype=float)
    scores = []
    for fold, (a, b) in enumerate(pairs, 1):
        folder = out / f'fold_{fold}'
        folder.mkdir()
        seed_all(settings['seed'])
        local = copy.deepcopy(settings)
        local['fold'] = fold
        write_json(folder / 'settings.json', local)
        train_data, val_data = (subset(tr, a), subset(tr, b))
        pd.DataFrame({'event_id': np.unique(val_data['event_id'])}).to_csv(folder / 'validation_events.csv', index=False)
        probability = validate_probability(module.train(train_data, val_data, local, folder, device))
        if len(probability) != len(b):
            raise ValueError('Prediction length does not match validation rows.')
        threshold = youden_threshold(val_data['y'], probability) if cohort == 'C1' else 0.5
        auc = float(roc_auc_score(val_data['y'], probability))
        write_json(folder / 'decision.json', {'threshold': threshold, 'validation_auc': auc})
        oof[b] = probability
        scores.append(auc)
        prediction_frame(val_data, probability, probability >= threshold).to_csv(folder / 'validation_predictions.csv', index=False)
    prediction_frame(tr, oof, oof >= 0.5).drop(columns='prediction').to_csv(out / 'out_of_fold_predictions.csv', index=False)
    run['status'] = 'complete'
    write_json(out / 'baseline_run.json', run)
    write_json(out / 'fit_summary.json', {'validation_auc_by_fold': scores, 'mean_validation_auc': float(np.mean(scores)), 'evaluation': 'internal training-fold validation; not an independent test'})
    return scores

def prediction_frame(data, probability, prediction):
    return pd.DataFrame({'event_id': data['event_id'], 'hour': data['hour'], 'label': data['y'], 'probability': probability, 'prediction': np.asarray(prediction, dtype=int)})

def predict(model, data_path, model_dir, output, device=None):
    data, root = (checked_data(data_path), Path(model_dir))
    run = read_json(root / 'baseline_run.json')
    if run['model'] != model or run['status'] != 'complete' or run['folds'] != 5:
        raise ValueError('Wrong model or incomplete five-fold run.')
    if data['feature_names'].tolist() != run['feature_names']:
        raise ValueError('Prediction feature names/order differ from training.')
    device = resolve_device(run['settings'].get('device', 'cpu'), device, cpu_only=model == 'random_forest')
    torch.set_num_threads(run['settings']['threads'])
    module = model_module(model)
    probabilities, decisions = ([], [])
    for fold in range(1, 6):
        folder = root / f'fold_{fold}'
        settings = read_json(folder / 'settings.json')
        p = validate_probability(module.predict(data['X'], settings, folder, device))
        if len(p) != len(data['y']):
            raise ValueError('Prediction length differs from input rows.')
        probabilities.append(p)
        decisions.append(p >= read_json(folder / 'decision.json')['threshold'])
    probability = np.mean(probabilities, axis=0)
    prediction = np.sum(decisions, axis=0) >= 3 if run['cohort'] == 'C1' else probability >= 0.5
    frame = prediction_frame(data, probability, prediction)
    for fold, p in enumerate(probabilities, 1):
        frame[f'probability_fold_{fold}'] = p
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame

def main(model):
    parser = argparse.ArgumentParser(description=f'{model}: standalone paper comparator, no distillation')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('train')
    p.add_argument('--cohort', required=True, choices=['C1', 'C2'])
    p.add_argument('--data', required=True, help='Directory containing prepared train.npz and optional val.npz')
    p.add_argument('--output', required=True, help='New or empty model directory')
    p.add_argument('--config', default=str(ROOT / 'configs/baselines.json'))
    p.add_argument('--device', default=None, choices=['cpu', 'cuda'], help='Override the configured device')
    p = commands.add_parser('predict')
    p.add_argument('--data', required=True, help='Prepared validation/test .npz')
    p.add_argument('--models', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default=None, choices=['cpu', 'cuda'], help='Override the configured device')
    p = commands.add_parser('evaluate')
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--prefix', default='val')
    p.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.command == 'train':
        train(model, args.cohort, args.data, args.output, args.config, args.device)
    elif args.command == 'predict':
        predict(model, args.data, args.models, args.output, args.device)
    else:
        evaluate_csv(args.predictions, args.output, args.prefix, args.bootstrap)
        path = Path(args.output) / f'{args.prefix}_DCA_data.csv'
        pd.read_csv(path).rename(columns={'AICI_Net_Benefit': f'{model}_Net_Benefit'}).to_csv(path, index=False)
