# Train XGBoost teachers, distill LF-Transformer students, and generate predictions.
from pathlib import Path
import copy
from .devices import resolve_device
import shutil
import numpy as np
import pandas as pd
import torch
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from .data import ROOT, TASKS, load_module, read_json, write_json, load_arrays, subset, make_folds, assert_separate
from .training import seed_all, build_model, train_epoch, student_values, loader_for, predict_model

def select_features(teacher, n_features, count):
    importance = teacher.get_score(importance_type='weight')
    scores = np.asarray([importance.get(f'f{i}', 0.0) for i in range(n_features)])
    return np.argsort(scores, kind='stable')[-min(count, n_features):]

def teacher_score(teacher, x, kind):
    if kind not in ('probability', 'margin'):
        raise ValueError('teacher_score must be probability or margin')
    return teacher.predict(xgb.DMatrix(x), output_margin=kind == 'margin', iteration_range=(0, teacher.best_iteration + 1))

def fit_teacher(train, val, cfg, out, device):
    best = {'auc': -np.inf}
    xt = xgb.DMatrix(train['X'], label=train['y'])
    xv = xgb.DMatrix(val['X'], label=val['y'])
    spec = cfg['teacher']
    source = ROOT / TASKS[cfg['cohort']] / 'xgb_model_train.py'
    space = load_module(source, f"teacher_candidates_{cfg['cohort']}").SEARCH_SPACE

    def objective(trial):
        seed_all(cfg['seed'])
        params = {'objective': 'binary:logistic', 'eval_metric': 'error', 'tree_method': 'hist', 'device': device, 'seed': cfg['seed'], 'nthread': cfg.get('threads', 4)}
        params.update({name: trial.suggest_categorical(name, values) for name, values in space.items()})
        model = xgb.train(params, xt, num_boost_round=spec['rounds'], evals=[(xv, 'val')], early_stopping_rounds=spec['patience'], verbose_eval=False)
        auc = float(roc_auc_score(val['y'], model.predict(xv, iteration_range=(0, model.best_iteration + 1))))
        if auc > best['auc']:
            best.update(auc=auc, params=params)
            model.save_model(out / 'teacher.json')
        return auc
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=cfg['seed']))
    study.optimize(objective, n_trials=spec['trials'])
    study.trials_dataframe().to_csv(out / 'teacher_trials.csv', index=False)
    model = xgb.Booster()
    model.load_model(out / 'teacher.json')
    write_json(out / 'teacher_search.json', {'best_validation_auc': best['auc'], 'params': best['params'], 'cohort': cfg['cohort'], 'search_space': space, 'search_space_source': source.relative_to(ROOT).as_posix(), 'selection_metric': 'validation_auc', 'search_space_status': 'Candidates restored from original cohort BO_TPE functions; not a historical run replay'})
    return model

def fit_student(train, val, teacher, indices, cfg, out, device):
    spec = cfg['student']
    tx = student_values(train['X'], indices)
    vx = student_values(val['X'], indices)
    score = teacher_score(teacher, train['X'], cfg['teacher_score'])
    counts = np.bincount(train['y'], minlength=2)
    if (counts == 0).any():
        raise ValueError('Training requires both classes.')
    weights = torch.tensor(len(train['y']) / (2.0 * counts), dtype=torch.float32, device=device)
    global_best = {'auc': -np.inf}

    def objective(trial):
        seed_all(cfg['seed'])
        conf = dict(spec['model'])
        for name, choices in spec.get('architecture_search', {}).items():
            conf[name] = trial.suggest_categorical(name, choices)
        lr = trial.suggest_float('lr', *spec['lr_range'], log=True)
        decay = trial.suggest_float('weight_decay', *spec['weight_decay_range'], log=True)
        model = build_model(cfg['cohort'], len(indices), conf, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
        loader = loader_for(tx, train['y'], score, conf['batch_size'], cfg['seed'])
        best = -np.inf
        stale = 0
        history = []
        for epoch in range(spec['epochs']):
            metrics = train_epoch(model, optimizer, loader, device, conf['batch_size'], weights)
            probs = predict_model(model, vx, conf['batch_size'], device)
            value = float(roc_auc_score(val['y'], probs))
            metrics.update(epoch=epoch, validation_auc=value)
            history.append(metrics)
            trial.report(value, epoch)
            if value > best:
                best = value
                stale = 0
            else:
                stale += 1
            if value > global_best['auc']:
                global_best.update(auc=value, config=conf, epoch=epoch, trial=trial.number)
                torch.save({'state_dict': model.state_dict(), 'model_config': conf, 'cohort': cfg['cohort'], 'feature_count': len(indices), 'teacher_score': cfg['teacher_score'], 'temperature': 4.0, 'cosine_threshold': 0.9, 'causal_mask': False}, out / 'student.pt')
                np.save(out / 'validation_probability.npy', probs)
            if stale >= spec['patience']:
                break
        pd.DataFrame(history).to_csv(out / f'student_trial_{trial.number}_history.csv', index=False)
        return best
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=cfg['seed']))
    study.optimize(objective, n_trials=spec['trials'])
    study.trials_dataframe().to_csv(out / 'student_trials.csv', index=False)
    write_json(out / 'student_search.json', global_best)
    return np.load(out / 'validation_probability.npy')

def youden_threshold(labels, prob):
    fpr, tpr, thresholds = roc_curve(labels, prob)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    return float(thresholds[valid][np.argmax((tpr - fpr)[valid])])

def fit(config, data_dir, output, device=None, stage='all'):
    cfg = copy.deepcopy(read_json(config) if not isinstance(config, dict) else config)
    teacher_device = resolve_device(cfg['teacher'].get('device', 'cpu'), device) if stage in ('all', 'teacher') else 'cpu'
    student_device = resolve_device(cfg['student'].get('device', 'cpu'), device) if stage != 'teacher' else None
    if stage in ('all', 'teacher'):
        cfg['teacher']['device'] = teacher_device
    if stage != 'teacher':
        cfg['student']['device'] = student_device
    data_dir = Path(data_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = load_arrays(data_dir / 'train.npz')
    if cfg['cohort'] in ('C1', 'C2'):
        pairs = [(subset(train, a), subset(train, b)) for a, b in make_folds(train, cfg['seed'])]
    else:
        val = load_arrays(data_dir / 'val.npz')
        assert_separate(train, val)
        pairs = [(train, val)]
    if cfg['cohort'] in ('C1', 'C2') and (data_dir / 'val.npz').exists():
        assert_separate(train, load_arrays(data_dir / 'val.npz'))
    write_json(output / 'run_config.json', cfg)
    results = []
    for fold, (tr, va) in enumerate(pairs, 1):
        out = output / f'fold_{fold}'
        out.mkdir(exist_ok=True)
        pd.DataFrame({'event_id': np.unique(va['event_id'])}).to_csv(out / 'validation_events.csv', index=False)
        write_json(out / 'feature_names.json', train['feature_names'].tolist())
        if stage in ('all', 'teacher'):
            teacher = fit_teacher(tr, va, cfg, out, teacher_device)
            count = cfg.get('selected_features') or int(np.ceil(tr['X'].shape[1] / 100.0) * 50)
            indices = select_features(teacher, tr['X'].shape[1], count)
            np.save(out / 'top_features.npy', indices)
        else:
            teacher = xgb.Booster()
            teacher.load_model(out / 'teacher.json')
            teacher.set_param({'device': 'cpu'})
            indices = np.load(out / 'top_features.npy')
        if stage == 'teacher':
            continue
        prob = fit_student(tr, va, teacher, indices, cfg, out, student_device)
        threshold = youden_threshold(va['y'], prob)
        write_json(out / 'decision.json', {'threshold': threshold, 'rule': 'probability >= threshold', 'source': 'training fold holdout' if cfg['cohort'] in ('C1', 'C2') else '2023 validation', 'validation_auc': float(roc_auc_score(va['y'], prob))})
        pd.DataFrame({'event_id': va['event_id'], 'hour': va['hour'], 'label': va['y'], 'probability': prob, 'prediction': (prob >= threshold).astype(int), 'fold': fold}).to_csv(out / 'validation_predictions.csv', index=False)
        results.append(float(roc_auc_score(va['y'], prob)))
    if results:
        write_json(output / 'fit_summary.json', {'folds': len(results), 'validation_auc_by_fold': results, 'mean_validation_auc': float(np.mean(results)), 'evaluation': 'five-fold validation mean' if len(results) == 5 else 'temporal validation used for selection'})
    return results

def predict(data_path, model_dir, output, device=None):
    data = load_arrays(data_path)
    root = Path(model_dir)
    cfg = read_json(root / 'run_config.json')
    n = 5 if cfg['cohort'] in ('C1', 'C2') else 1
    device = resolve_device(cfg['student'].get('device', 'cpu'), device)
    probabilities = []
    decisions = []
    for fold in range(1, n + 1):
        folder = root / f'fold_{fold}'
        if data['feature_names'].tolist() != read_json(folder / 'feature_names.json'):
            raise ValueError('Prediction feature schema differs from training.')
        payload = torch.load(folder / 'student.pt', map_location=device, weights_only=True)
        if payload['cohort'] != cfg['cohort']:
            raise ValueError('Wrong task checkpoint')
        conf = payload['model_config']
        indices = np.load(folder / 'top_features.npy')
        model = build_model(cfg['cohort'], payload['feature_count'], conf, device)
        model.load_state_dict(payload['state_dict'])
        p = predict_model(model, student_values(data['X'], indices), conf['batch_size'], device)
        probabilities.append(p)
        decisions.append(p >= read_json(folder / 'decision.json')['threshold'])
    probability = np.mean(probabilities, axis=0)
    prediction = (np.sum(decisions, axis=0) >= 3).astype(int) if cfg['cohort'] == 'C1' else (probability >= 0.5).astype(int) if cfg['cohort'] == 'C2' else decisions[0].astype(int)
    frame = pd.DataFrame({'event_id': data['event_id'], 'hour': data['hour'], 'label': data['y'], 'probability': probability, 'prediction': prediction})
    for i, p in enumerate(probabilities, 1):
        frame[f'probability_fold_{i}'] = p
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog='python -m aici.pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fit', help='Fit five folds or one temporal train/val model with Optuna')
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default=None, choices=['cpu', 'cuda'], help='Override configured devices; omit to use configuration')
    p.add_argument('--stage', default='all', choices=['all', 'teacher', 'student'])
    p = sub.add_parser('predict', help='Predict with saved configuration and task-specific weights')
    p.add_argument('--data', required=True)
    p.add_argument('--models', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default=None, choices=['cpu', 'cuda'], help='Override configured devices; omit to use configuration')
    args = parser.parse_args(argv)
    if args.command == 'fit':
        return fit(args.config, args.data, args.output, args.device, args.stage)
    return predict(args.data, args.models, args.output, args.device)
if __name__ == '__main__':
    main()
