# Prepare cohort data, input manifests, and feature arrays.
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from .preprocessing import fit_iqr_bounds, apply_iqr_bounds, hourly_records, sepsis_labels, mortality_label, icu_stay_label
ROOT = Path(__file__).resolve().parents[1]
TASKS = {'C1': 'C1_sepsis_onset_prediction', 'C2': 'C2_sepsis_onset_prediction', 'C3': 'C3_in_hospital_mortality_prediction', 'C4': 'C4_prolonged_ICU_stay_prediction'}

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def feature_module(cohort):
    folder = ROOT / TASKS[cohort]
    sys.path.insert(0, str(folder))
    try:
        return load_module(folder / 'feature_engineering.py', f'features_{cohort}')
    finally:
        sys.path.pop(0)

def raw_columns(cohort, schema):
    if not schema or not schema.get('raw'):
        raise ValueError(f'{cohort} raw input needs a verified --schema.')
    columns = schema['raw']
    if len(set(columns)) != len(columns):
        raise ValueError('Raw variables must be unique.')
    groups = schema.get('continuous', []) + schema.get('intermittent', [])
    if len(set(groups)) != len(groups) or not set(groups) <= set(columns):
        raise ValueError('Missingness groups must be disjoint subsets of raw variables.')
    return list(schema['raw'])

def read_stay(path):
    path = Path(path)
    if path.suffix.lower() == '.xlsx':
        return pd.read_excel(path)
    return pd.read_csv(path, sep='|' if path.suffix.lower() == '.psv' else ',')

def read_manifest(path):
    manifest = pd.read_csv(path, dtype={'event_id': str, 'subject_id': str})
    if not {'file', 'event_id', 'split'} <= set(manifest):
        raise ValueError('Manifest needs file,event_id,split.')
    if manifest[['file', 'event_id', 'split']].isna().any().any() or manifest.event_id.duplicated().any():
        raise ValueError('Each manifest row needs a file, a unique event_id and a split.')
    if not set(manifest.split) <= {'train', 'val', 'test'}:
        raise ValueError('split must be train, val or test')
    files = [(Path(path).resolve().parent / f).resolve() for f in manifest.file]
    if len(set(files)) != len(files):
        raise ValueError('The same file cannot be listed as multiple stays.')
    return manifest

def _mapped_observations(frame, item_map):
    frame = frame.rename(columns={'charttime': 'result_time'}).copy()
    if 'variable' not in frame:
        if 'itemid' not in frame or item_map is None:
            raise ValueError('Long input needs variable, or itemid plus --item-map.')
        unknown = set(frame.itemid.astype(str)) - set(item_map)
        if unknown:
            raise ValueError(f'Unmapped itemid values: {sorted(unknown)}')
        rules = frame.itemid.astype(str).map(item_map)
        frame['variable'] = rules.map(lambda rule: rule['variable'])
        frame['value'] = pd.to_numeric(frame['value'], errors='raise') * rules.map(lambda rule: rule.get('multiplier', 1.0)).astype(float) + rules.map(lambda rule: rule.get('offset', 0.0)).astype(float)
    if not {'result_time', 'variable', 'value'} <= set(frame):
        raise ValueError('Long input needs result_time,variable,value.')
    frame['value'] = pd.to_numeric(frame['value'], errors='raise')
    if frame.empty or frame.result_time.isna().any() or frame.variable.isna().any():
        raise ValueError('Long input contains no observations or has missing timestamps/variables.')
    if np.isinf(frame.value).any():
        raise ValueError('Long input contains infinite measurements.')
    return frame

def build_hourly(cohort, manifest_path, output, schema_path, item_map_path=None):
    if cohort not in ('C2', 'C3', 'C4'):
        raise ValueError('C1 already provides hourly PSV records.')
    source = Path(manifest_path).resolve()
    manifest = read_manifest(source)
    if cohort == 'C2' and 'diagnosis_time' not in manifest:
        raise ValueError('C2 needs diagnosis_time in the manifest; blank explicitly denotes no sepsis diagnosis.')
    schema = read_json(schema_path)
    raw = raw_columns(cohort, schema)
    item_map = read_json(item_map_path) if item_map_path else None
    if item_map is not None:
        if not item_map or any((not isinstance(rule, dict) or schema.get('rename', {}).get(rule.get('variable'), rule.get('variable')) not in raw for rule in item_map.values())):
            raise ValueError('Item map needs itemid -> {variable, multiplier, offset} with known variables.')
    output = Path(output)
    (output / 'hourly').mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(manifest.to_dict('records'), 1):
        observations = read_stay(source.parent / record['file'])
        for identity in ['event_id', 'stay_id', 'subject_id']:
            if identity in observations and observations[identity].nunique(dropna=False) > 1:
                raise ValueError('Each long input file must contain only one stay/subject.')
        observations = _mapped_observations(observations, item_map)
        observations['variable'] = observations['variable'].replace(schema.get('rename', {}))
        unknown = set(observations.variable) - set(raw)
        if unknown:
            raise ValueError(f"Unrecognized variables in {record['file']}: {sorted(unknown)}")
        origin = record.get('origin_time')
        if pd.isna(origin):
            origin = pd.to_datetime(observations.result_time).min().floor('h')
        wide = hourly_records(observations, origin).set_index('hour').reindex(columns=raw).reset_index()
        if cohort == 'C2':
            diagnosis = record.get('diagnosis_time')
            if pd.notna(diagnosis):
                diagnosis_hour = (pd.Timestamp(diagnosis) - pd.Timestamp(origin)).total_seconds() / 3600
            else:
                diagnosis_hour = None
            wide['label'] = sepsis_labels(wide.hour, diagnosis_hour)
        elif cohort == 'C3':
            outcome = record.get('discharge_outcome')
            label = mortality_label(outcome) if pd.notna(outcome) else record.get('label')
            if pd.isna(label):
                raise ValueError('C3 needs adjudicated discharge_outcome or binary label.')
            wide['label'] = label
        else:
            duration = record.get('icu_total_hours')
            label = icu_stay_label(float(duration)) if pd.notna(duration) else record.get('label')
            if pd.isna(label):
                raise ValueError('C4 needs icu_total_hours or binary label.')
            wide['label'] = label
        if not np.isin(wide.label, [0, 1]).all():
            raise ValueError('Outcome labels must be binary 0/1.')
        name = f'hourly/stay_{index:06d}.csv'
        wide.to_csv(output / name, index=False)
        rows.append({'file': name, 'event_id': record['event_id'], 'split': record['split'], 'subject_id': record.get('subject_id')})
    pd.DataFrame(rows).to_csv(output / 'manifest.csv', index=False)
    write_json(output / 'hourly_build.json', {'cohort': cohort, 'events': len(rows), 'time_basis': 'recorded result_time; hour is [origin + h, origin + h + 1 hour)', 'aggregation': 'mean of numeric results in the same hour', 'raw_columns': raw, 'item_map_used': bool(item_map)})
    return output / 'manifest.csv'

def fit_iqr_from_manifest(cohort, manifest_path, output, schema_path=None, min_hours=8):
    if cohort not in TASKS:
        raise ValueError(f'Unknown cohort {cohort}')
    if min_hours < 1:
        raise ValueError('min_hours must be positive.')
    source = Path(manifest_path).resolve()
    manifest = read_manifest(source)
    schema = read_json(schema_path or ROOT / 'configs' / f'{cohort}_schema.json')
    columns = raw_columns(cohort, schema)
    values = []
    events = 0
    for record in manifest.loc[manifest.split == 'train'].to_dict('records'):
        frame = read_stay(source.parent / record['file']).rename(columns=(schema or {}).get('rename', {}))
        if len(frame) < min_hours:
            continue
        missing = set(columns) - set(frame)
        if missing:
            raise ValueError(f"Missing raw columns in {record['file']}: {sorted(missing)}")
        values.append(frame[columns].apply(pd.to_numeric, errors='raise'))
        events += 1
    if not values:
        raise ValueError('No eligible training stays to estimate IQR bounds.')
    bounds = fit_iqr_bounds(pd.concat(values, ignore_index=True), columns)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bounds.to_csv(output, index=False)
    write_json(output.with_suffix('.json'), {'cohort': cohort, 'training_events': events, 'training_rows': sum((len(v) for v in values)), 'min_hours': min_hours, 'columns': columns, 'rule': 'Q1 - 1.5*IQR; Q3 + 1.5*IQR; outside values become missing'})
    return bounds

def prepare(cohort, manifest_path, output, schema_path=None, engineered=False, min_hours=8, iqr_bounds=None):
    if cohort not in TASKS:
        raise ValueError(f'Unknown cohort {cohort}')
    if min_hours < 1:
        raise ValueError('min_hours must be positive.')
    source = Path(manifest_path).resolve()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(source)
    schema = read_json(schema_path or ROOT / 'configs' / f'{cohort}_schema.json')
    if engineered and iqr_bounds:
        raise ValueError('Engineered features must already be cleaned; omit --iqr-bounds.')
    if not engineered and (not iqr_bounds):
        raise ValueError('Raw input requires training-only --iqr-bounds; use fit-iqr first.')
    bounds = pd.read_csv(iqr_bounds) if iqr_bounds else None
    raw = raw_columns(cohort, schema) if not engineered else None
    if bounds is not None and (not {'feature', 'lower', 'upper'} <= set(bounds) or bounds['feature'].tolist() != raw or bounds.feature.duplicated().any()):
        raise ValueError('IQR bound columns or order differ from the raw schema.')
    module = None if engineered else feature_module(cohort)
    frames = {}
    names = None
    excluded = []
    for record in manifest.to_dict('records'):
        path = (source.parent / record['file']).resolve()
        frame = read_stay(path)
        hour = pd.to_numeric(frame['hour'], errors='raise').to_numpy() if 'hour' in frame else np.arange(len(frame))
        if not np.isfinite(hour).all() or np.any(hour < 0) or np.any(hour != np.floor(hour)) or np.any(np.diff(hour) != 1):
            raise ValueError('Input must be one ordered, consecutive hourly row per stay; reindex missing hours first.')
        if len(frame) < min_hours:
            excluded.append({'event_id': record['event_id'], 'rows': len(frame), 'reason': 'fewer than min_hours'})
            continue
        label_name = next((k for k in ['label', 'labels', 'SepsisLabel'] if k in frame), None)
        if label_name:
            labels = frame[label_name].to_numpy()
        elif cohort in ('C3', 'C4') and pd.notna(record.get('label')):
            labels = np.full(len(frame), record['label'])
        else:
            raise ValueError('Each file needs a label column or a manifest-level outcome label.')
        if not np.isin(labels, [0, 1]).all():
            raise ValueError('Labels must be binary 0/1.')
        if cohort in ('C3', 'C4') and len(np.unique(labels)) != 1:
            raise ValueError('C3/C4 use the same final outcome label across a stay.')
        if not engineered:
            frame = frame.rename(columns=(schema or {}).get('rename', {})).copy()
            if frame.columns.duplicated().any():
                raise ValueError('Duplicate columns after variable renaming.')
            missing = set(raw) - set(frame)
            if missing:
                raise ValueError(f'Missing configured clinical columns: {sorted(missing)}')
            frame[raw] = frame[raw].apply(pd.to_numeric, errors='raise')
            if np.isinf(frame[raw].to_numpy(dtype=float)).any():
                raise ValueError('Infinite raw measurements found.')
            frame = apply_iqr_bounds(frame, bounds)
        if engineered:
            features = frame.drop(columns=['hour', 'label', 'labels', 'SepsisLabel', 'event_id', 'subject_id', 'stay_id', 'split', 'diagnosis_time', 'diagnosis_hour', 'sofa_time', 'discharge_outcome', 'icu_total_hours', 'result_time', 'charttime', 'origin_time'], errors='ignore')
        elif cohort == 'C1':
            clean = frame[raw].copy()
            clean['SepsisLabel'] = labels
            features, _ = module.feature_extraction(clean, raw, schema['continuous'], schema['intermittent'])
        else:
            raw = schema['raw']
            continuous = schema['continuous']
            intermittent = schema['intermittent']
            clean = frame[raw].copy()
            clean['label'] = labels
            result = module.feature_extraction(clean, raw, continuous, intermittent)
            features = result[0] if isinstance(result, tuple) else result
        if features.columns.duplicated().any():
            raise ValueError('Feature names must be unique.')
        if names is None:
            names = list(features.columns)
        if list(features.columns) != names:
            raise ValueError('All files must have the same feature column order.')
        values = features.to_numpy(dtype=np.float32)
        if np.isinf(values).any():
            raise ValueError('Infinite features found; supply the study-specific upstream bounds/rules.')
        entry = (values, labels.astype(np.int64), np.repeat(str(record['event_id']), len(frame)), hour, np.repeat(str(record['subject_id']) if pd.notna(record.get('subject_id')) else str(record['event_id']), len(frame)))
        frames.setdefault(record['split'], []).append(entry)
    if not frames:
        raise ValueError('No eligible input files.')
    for split, records in frames.items():
        arrays = [np.concatenate([r[i] for r in records]) for i in range(5)]
        np.savez_compressed(out / f'{split}.npz', X=arrays[0], y=arrays[1], event_id=arrays[2], hour=arrays[3], subject_id=arrays[4], feature_names=np.asarray(names))
    write_json(out / 'prepared.json', {'cohort': cohort, 'min_hours': min_hours, 'includes_first_seven_hours': True, 'iqr_bounds_applied': bool(iqr_bounds), 'iqr_bounds_file': Path(iqr_bounds).name if iqr_bounds else None, 'feature_names': names, 'rows': {s: sum((len(r[1]) for r in rs)) for s, rs in frames.items()}, 'excluded': excluded})
    return out

def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k] for k in z.files}
    for key in ['X', 'y', 'event_id', 'hour', 'feature_names']:
        if key not in result:
            raise ValueError(f'Missing {key} in {path}')
    if result['X'].ndim != 2 or len(result['X']) != len(result['y']):
        raise ValueError('Invalid X/y shape')
    if not np.isin(result['y'], [0, 1]).all():
        raise ValueError('Labels must be binary')
    return result

def subset(data, indices):
    return {k: v if k == 'feature_names' else v[indices] for k, v in data.items()}

def make_folds(data, seed=3407):
    from sklearn.model_selection import StratifiedKFold
    events = np.unique(data['event_id'])
    labels = np.asarray([data['y'][data['event_id'] == e].max() for e in events])
    if min(np.bincount(labels, minlength=2)) < 5:
        raise ValueError('Five folds need at least five stays in each outcome group.')
    result = []
    for train, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(events, labels):
        a = np.flatnonzero(np.isin(data['event_id'], events[train]))
        b = np.flatnonzero(np.isin(data['event_id'], events[val]))
        result.append((a, b))
    return result

def assert_separate(*datasets):
    for i, a in enumerate(datasets):
        for b in datasets[i + 1:]:
            if set(a['event_id']) & set(b['event_id']):
                raise ValueError('The same stay occurs in multiple splits.')
            if not np.array_equal(a['feature_names'], b['feature_names']):
                raise ValueError('Split feature schemas differ.')

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog='python -m aici.data')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build-hourly', help='Result-time long files -> per-stay hourly wide files')
    p.add_argument('--cohort', required=True, choices=['C2', 'C3', 'C4'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--schema', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--item-map', help='Local itemid -> variable/unit conversion JSON, for C2 SQL exports')
    p = sub.add_parser('prepare', help='Hourly records or engineered CSVs -> per-split NPZ')
    p.add_argument('--cohort', required=True, choices=['C1', 'C2', 'C3', 'C4'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--schema')
    p.add_argument('--engineered', action='store_true')
    p.add_argument('--iqr-bounds', help='CSV estimated from eligible training stays by fit-iqr; required for raw input')
    p.add_argument('--min-hours', type=int, default=8)
    p = sub.add_parser('fit-iqr', help='Estimate raw-variable IQR limits from eligible training stays only')
    p.add_argument('--cohort', required=True, choices=['C1', 'C2', 'C3', 'C4'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--schema')
    p.add_argument('--min-hours', type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == 'build-hourly':
        return build_hourly(args.cohort, args.manifest, args.output, args.schema, args.item_map)
    if args.command == 'prepare':
        return prepare(args.cohort, args.manifest, args.output, args.schema, args.engineered, args.min_hours, args.iqr_bounds)
    return fit_iqr_from_manifest(args.cohort, args.manifest, args.output, args.schema, args.min_hours)
if __name__ == '__main__':
    main()
