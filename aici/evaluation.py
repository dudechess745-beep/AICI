# Calculate prediction metrics and export evaluation tables.
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from .data import write_json

def validate_probability(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities must be finite and within [0,1]; raw SOFA scores are not probabilities.')
    return p

def calibration(labels, prob, bins=20):
    prob = validate_probability(prob)
    labels = np.asarray(labels)
    index = np.minimum(np.floor(prob * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        mask = index == i
        if mask.any():
            rows.append({'Mean_Predicted_Probability': float(prob[mask].mean()), 'True_Probability': float(labels[mask].mean()), 'N': int(mask.sum()), 'Bin': i})
    return pd.DataFrame(rows)

def decision_curve(labels, prob, sofa_probability=None):
    labels = np.asarray(labels, dtype=int)
    prob = validate_probability(prob)
    n = len(labels)
    if not n:
        raise ValueError('Empty predictions')
    sofa = None if sofa_probability is None else validate_probability(sofa_probability)
    thresholds = np.linspace(0, 0.99, 100)
    rows = []
    for t in thresholds:
        ratio = t / (1 - t)

        def nb(p):
            selected = p >= t
            return float(np.sum(selected & (labels == 1)) / n - np.sum(selected & (labels == 0)) / n * ratio)
        row = {'Threshold': t, 'AICI_Net_Benefit': nb(prob), 'Treat_All_Net_Benefit': float(labels.mean() - (1 - labels.mean()) * ratio), 'Treat_None_Net_Benefit': 0.0}
        if sofa is not None:
            row['SOFA_Net_Benefit'] = nb(sofa)
        rows.append(row)
    return pd.DataFrame(rows)

def evaluate_csv(predictions, output, prefix='val', bootstrap=1000, seed=3407):
    frame = pd.read_csv(predictions)
    y = frame['label'].to_numpy()
    p = validate_probability(frame['probability'])
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Evaluation requires binary labels and both classes.')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    curve = calibration(y, p)
    curve.to_csv(out / f'{prefix}_calibration_data.csv', index=False)
    curve.to_csv(out / f'{prefix}_calibration_curve_data.csv', index=False)
    dca = decision_curve(y, p, frame['sofa_probability'].to_numpy() if 'sofa_probability' in frame else None)
    dca.to_csv(out / f'{prefix}_DCA_data.csv', index=False)
    value = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            replicates.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.quantile(replicates, [0.025, 0.975]).tolist() if replicates else [None, None]
    fpr, tpr, thresholds = roc_curve(y, p)
    roc = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds, 'AUC': value, 'AUC_CI_Lower': lo, 'AUC_CI_Upper': hi})
    roc.to_csv(out / f'roc_data_{prefix}.csv', index=False)
    pred = frame['prediction'].to_numpy() if 'prediction' in frame else (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {'auc': value, 'accuracy': float((tn + tp) / len(y)), 'sensitivity': float(tp / (tp + fn)), 'specificity': float(tn / (tn + fp)), 'rows': len(y), 'bootstrap_unit': 'hourly row (preserved for review v1)', 'ci_lower': lo, 'ci_upper': hi, 'sofa_probability_provided': 'sofa_probability' in frame}
    write_json(out / f'{prefix}_metrics.json', metrics)
    return metrics

def fit_sofa(train_csv, model_path):
    from sklearn.linear_model import LogisticRegression
    data = pd.read_csv(train_csv)
    values = data['sofa_score'].to_numpy(dtype=float)
    y = data['label'].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError('SOFA score must be finite')
    model = LogisticRegression(random_state=3407).fit(values.reshape(-1, 1), y)
    write_json(model_path, {'coefficient': float(model.coef_[0, 0]), 'intercept': float(model.intercept_[0]), 'fit_split': 'train', 'description': 'new optional logistic SOFA mapping; do not claim identical historical calibration'})

def apply_sofa(predictions, model_path, output):
    from scipy.special import expit
    from .data import read_json
    data = pd.read_csv(predictions)
    model = read_json(model_path)
    if model.get('fit_split') != 'train':
        raise ValueError('SOFA mapper must be fitted on train')
    data['sofa_probability'] = expit(data['sofa_score'] * model['coefficient'] + model['intercept'])
    validate_probability(data['sofa_probability'])
    data.to_csv(output, index=False)

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog='python -m aici.evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('evaluate', help='Predictions -> ROC, calibration and DCA CSVs')
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--prefix', default='val')
    p.add_argument('--bootstrap', type=int, default=1000)
    p = sub.add_parser('fit-sofa')
    p.add_argument('--train', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('apply-sofa')
    p.add_argument('--predictions', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'evaluate':
        result = evaluate_csv(args.predictions, args.output, args.prefix, args.bootstrap)
        print(result)
        return result
    if args.command == 'fit-sofa':
        return fit_sofa(args.train, args.output)
    return apply_sofa(args.predictions, args.model, args.output)
if __name__ == '__main__':
    main()
