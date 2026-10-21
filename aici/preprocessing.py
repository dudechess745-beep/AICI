# Aggregate hourly records, estimate IQR bounds, and construct outcome labels.
import numpy as np
import pandas as pd

def hourly_records(observations, origin):
    frame = observations.copy()
    frame['result_time'] = pd.to_datetime(frame['result_time'])
    frame['hour'] = np.floor((frame['result_time'] - pd.Timestamp(origin)).dt.total_seconds() / 3600).astype(int)
    if (frame.hour < 0).any():
        raise ValueError('Result timestamp precedes the configured observation origin.')
    wide = frame.pivot_table(index='hour', columns='variable', values='value', aggfunc='mean')
    return wide.reindex(range(int(frame.hour.max()) + 1)).rename_axis('hour').reset_index()

def fit_iqr_bounds(training_rows, columns):
    if np.isinf(training_rows[columns].to_numpy(dtype=float)).any():
        raise ValueError('Infinite raw measurements must be resolved before fitting IQR bounds.')
    q1 = training_rows[columns].quantile(0.25)
    q3 = training_rows[columns].quantile(0.75)
    iqr = q3 - q1
    return pd.DataFrame({'feature': columns, 'lower': (q1 - 1.5 * iqr).to_numpy(), 'upper': (q3 + 1.5 * iqr).to_numpy()})

def apply_iqr_bounds(rows, bounds):
    result = rows.copy()
    for b in bounds.itertuples(index=False):
        if b.feature not in result:
            raise ValueError(f'Missing bounded feature {b.feature}')
        if pd.isna(b.lower) and pd.isna(b.upper):
            result[b.feature] = np.nan
            continue
        if not np.isfinite([b.lower, b.upper]).all() or b.lower > b.upper:
            raise ValueError(f'Invalid IQR limits for {b.feature}')
        x = result[b.feature]
        result.loc[(x < b.lower) | (x > b.upper), b.feature] = np.nan
    return result

def sepsis_labels(hours, diagnosis_hour=None):
    hours = np.asarray(hours, dtype=float)
    return np.zeros(len(hours), dtype=np.int64) if diagnosis_hour is None else (hours >= float(diagnosis_hour) - 6.0).astype(np.int64)

def mortality_label(discharge_outcome):
    labels = {'death': 1, 'discharge_against_medical_advice': 1, 'survived': 0}
    if discharge_outcome not in labels:
        raise ValueError('Use adjudicated death/discharge_against_medical_advice/survived.')
    return labels[discharge_outcome]

def icu_stay_label(total_hours):
    if not np.isfinite(total_hours) or total_hours < 0:
        raise ValueError('Invalid total ICU duration')
    return int(total_hours >= 24.0)
