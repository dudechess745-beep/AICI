# Construct clinical measurements, window statistics, and missing data features.
import os
import pandas as pd
import numpy as np
from itertools import chain
from function import save_pickle, load_pickle
from tqdm import tqdm
from pathlib import Path
import json
_DEFAULT_SCHEMA = json.loads((Path(__file__).resolve().parents[1] / 'configs/C1_schema.json').read_text(encoding='utf-8'))
use_raw_index = [name for name in _DEFAULT_SCHEMA['raw'] if name != 'FiO2']
con_index = _DEFAULT_SCHEMA['continuous']
sep_index = _DEFAULT_SCHEMA['intermittent']

def moving_average_with_nan(arr, n):
    result = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        non_nan_values = [x for x in arr[i::-1] if not np.isnan(x)]
        if non_nan_values:
            result[i] = np.mean(non_nan_values[:n])
    result = np.nan_to_num(result, nan=0.0)
    return result

def moving_std_with_nan(arr, n):
    result = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        non_nan_values = [x for x in arr[i::-1] if not np.isnan(x)]
        if non_nan_values:
            result[i] = np.std(non_nan_values[:n])
    result = np.nan_to_num(result, nan=0.0)
    return result

def nan_percentage_around_point(arr, n):
    result = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - n)
        end = min(len(arr), i + n + 1)
        neighbors = arr[start:end]
        nan_count = np.sum(np.isnan(neighbors))
        total_count = len(neighbors)
        if total_count > 0:
            result[i] = nan_count / total_count * 100
    return result

def CumulativeSumofNonNaN(arr):
    cumulative_sum = np.zeros_like(arr, dtype=float)
    non_nan_indices = ~np.isnan(arr)
    for i in range(1, len(arr)):
        if non_nan_indices[i]:
            cumulative_sum[i] = cumulative_sum[i - 1] + arr[i]
        else:
            cumulative_sum[i] = cumulative_sum[i - 1]
    return cumulative_sum

def cumulative_non_nan_average(data):
    cumulative_sum = np.zeros_like(data, dtype=float)
    count_non_nan = 0
    result = np.zeros_like(data, dtype=float)
    for i in range(len(data)):
        if not np.isnan(data[i]):
            count_non_nan += 1
            cumulative_sum[i] = cumulative_sum[i - 1] + data[i] if i > 0 else data[i]
        elif i > 0:
            cumulative_sum[i] = cumulative_sum[i - 1]
        if count_non_nan > 0:
            result[i] = cumulative_sum[i] / count_non_nan
    return result

def RatiotoPreviousNonNaN(data):
    ratios = np.full(data.shape, np.nan)
    prev_non_nan = None
    for i, val in enumerate(data):
        if not np.isnan(val):
            if prev_non_nan is not None:
                ratios[i] = val / prev_non_nan
            if val != 0:
                prev_non_nan = val
        elif not np.isnan(ratios[i - 1]) and i > 0:
            ratios[i] = ratios[i - 1]
    return ratios

def calculate_variance(data):
    variances = np.full(data.shape, np.nan)
    prev_non_nan = None
    for i, val in enumerate(data):
        if not np.isnan(val):
            if prev_non_nan is not None:
                diff = (val - prev_non_nan) / 2
                variances[i] = 2 * diff ** 2
            prev_non_nan = val
        elif not np.isnan(variances[i - 1]) and i > 0:
            variances[i] = variances[i - 1]
    return variances

def cumulative_max(data):
    cum_max = np.full(data.shape, np.nan)
    current_max = None
    for i, val in enumerate(data):
        if not np.isnan(val):
            if current_max is None or val > current_max:
                current_max = val
            cum_max[i] = current_max
        else:
            cum_max[i] = current_max
    return cum_max

def rolling_range(data, window_size=3):
    if window_size % 2 == 0:
        raise ValueError('Window size must be odd')
    half_window = window_size // 2
    rolling_ranges = np.full(data.shape, np.nan)
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window_data = data[start:end]
        window_data = window_data[~np.isnan(window_data)]
        if window_data.size > 0:
            rolling_ranges[i] = np.max(window_data) - np.min(window_data)
    return rolling_ranges

def missing_streak_forward(data):
    streaks = np.zeros(data.shape, dtype=int)
    streak_count = 0
    for i, val in enumerate(data):
        if np.isnan(val):
            streak_count += 1
        else:
            streak_count = 0
        streaks[i] = streak_count
    return streaks

def CumulativeMultiofNonNaN(arr):
    cumulative_sum = np.zeros_like(arr, dtype=float)
    non_nan_indices = ~np.isnan(arr)
    flag = 1
    for i in range(1, len(arr)):
        if non_nan_indices[i]:
            if flag:
                cumulative_sum[i] = (i + 1) * arr[i]
                flag = 0
            else:
                cumulative_sum[i] = cumulative_sum[i - 1] + (i + 1) * arr[i]
        else:
            cumulative_sum[i] = cumulative_sum[i - 1]
    return cumulative_sum

def count_missing_transitions(arr):
    arr = np.array(arr)
    transitions_record = np.zeros_like(arr, dtype=int)
    is_missing = np.isnan(arr)
    for i in range(1, len(arr)):
        if is_missing[i] != is_missing[i - 1]:
            transitions_record[i] = 1
    return transitions_record

def accumulate_missing_transitions(arr):
    arr = np.array(arr)
    is_missing = np.isnan(arr)
    transitions_accumulate = np.zeros(len(arr), dtype=int)
    change_count = 0
    for i in range(1, len(arr)):
        if is_missing[i] != is_missing[i - 1]:
            change_count += 1
        transitions_accumulate[i] = change_count
    return transitions_accumulate

def feature_informative_missingness(case, sep_columns):
    new_feature_count = 0
    temp_data = np.array(case)
    for sep_column in sep_columns:
        sep_data = np.array(case[sep_column])
        nan_pos = np.where(~np.isnan(sep_data))[0]
        interval_f1 = sep_data.copy()
        interval_f2 = sep_data.copy()
        if len(nan_pos) == 0:
            interval_f1[:] = 0
            temp_data = np.column_stack((temp_data, interval_f1))
            interval_f2[:] = -1
            temp_data = np.column_stack((temp_data, interval_f2))
        else:
            interval_f1[:nan_pos[0]] = 0
            for p in range(len(nan_pos) - 1):
                interval_f1[nan_pos[p]:nan_pos[p + 1]] = p + 1
            interval_f1[nan_pos[-1]:] = len(nan_pos)
            temp_data = np.column_stack((temp_data, interval_f1))
            interval_f2[:nan_pos[0]] = -1
            for q in range(len(nan_pos) - 1):
                length = nan_pos[q + 1] - nan_pos[q]
                for l in range(length):
                    interval_f2[nan_pos[q] + l] = l
            length = len(case) - nan_pos[-1]
            for l in range(length):
                interval_f2[nan_pos[-1] + l] = l
            temp_data = np.column_stack((temp_data, interval_f2))
        diff_f = sep_data.copy()
        diff_f = diff_f.astype(float)
        if len(nan_pos) <= 1:
            diff_f[:] = np.nan
            temp_data = np.column_stack((temp_data, diff_f))
        else:
            diff_f[:nan_pos[1]] = np.nan
            for p in range(1, len(nan_pos) - 1):
                diff_f[nan_pos[p]:nan_pos[p + 1]] = sep_data[nan_pos[p]] - sep_data[nan_pos[p - 1]]
            diff_f[nan_pos[-1]:] = sep_data[nan_pos[-1]] - sep_data[nan_pos[-2]]
            temp_data = np.column_stack((temp_data, diff_f))
        missing_duration = np.zeros(len(sep_data), dtype=float)
        missing_streak = 0
        for i, value in enumerate(sep_data):
            missing_streak = missing_streak + 1 if np.isnan(value) else 0
            missing_duration[i] = missing_streak
        temp_data = np.column_stack((temp_data, missing_duration))
        missing_cumulative_ratio = sep_data.copy()
        cumulative_missing_count = np.cumsum(np.isnan(missing_cumulative_ratio))
        missing_cumulative_ratio = cumulative_missing_count / (np.arange(len(sep_data)) + 1)
        temp_data = np.column_stack((temp_data, missing_cumulative_ratio))
        measurements_between_missing = sep_data.copy()
        if len(nan_pos) > 0:
            measurements_between_missing[:nan_pos[0]] = np.arange(1, nan_pos[0] + 1)
            for p in range(len(nan_pos) - 1):
                measurements_between_missing[nan_pos[p]:nan_pos[p + 1]] = np.arange(nan_pos[p + 1] - nan_pos[p])
            measurements_between_missing[nan_pos[-1]:] = np.arange(len(sep_data) - nan_pos[-1])
        else:
            measurements_between_missing[:] = np.arange(1, len(sep_data) + 1)
        temp_data = np.column_stack((temp_data, measurements_between_missing))
        missingornot = sep_data.copy()
        temp_data = np.column_stack((temp_data, missingornot))
        new_feature_count += 7
    new_columns = []
    for sep_column in sep_columns:
        new_columns.extend([f'{sep_column}_freq', f'{sep_column}_intev', f'{sep_column}_diff', f'{sep_column}_mDur', f'{sep_column}_mCum', f'{sep_column}_mbm', f'{sep_column}_mask'])
    temp_df = pd.DataFrame(temp_data, columns=case.columns.tolist() + new_columns[:new_feature_count])
    for sep_column in sep_columns:
        temp_df[f'{sep_column}_mask'] = temp_df[sep_column].notna().astype(int)
    return temp_df

def feature_slide_window(temp, feature_names):
    con_index = list(range(len(feature_names)))
    sepdata = temp.loc[:, feature_names].to_numpy()
    max_values = [[0 for col in range(len(sepdata))] for row in range(len(con_index))]
    min_values = [[0 for col in range(len(sepdata))] for row in range(len(con_index))]
    mean_values = [[0 for col in range(len(sepdata))] for row in range(len(con_index))]
    median_values = [[0 for col in range(len(sepdata))] for row in range(len(con_index))]
    std_values = [[0 for col in range(len(sepdata))] for row in range(len(con_index))]
    diff_std_values = [[0 for col in range(len(sepdata))] for row in range(len(con_index))]
    for i in range(len(sepdata)):
        if i < 6:
            win_data = sepdata[0:i + 1]
            for ii in range(6 - i):
                win_data = np.vstack((win_data, sepdata[i]))
        else:
            win_data = sepdata[i - 6:i + 1]
        for j in range(len(con_index)):
            dat = win_data[:, j]
            if len(np.where(~np.isnan(dat))[0]) == 0:
                max_values[j][i] = np.nan
                min_values[j][i] = np.nan
                mean_values[j][i] = np.nan
                median_values[j][i] = np.nan
                std_values[j][i] = np.nan
                diff_std_values[j][i] = np.nan
            else:
                max_values[j][i] = np.nanmax(dat)
                min_values[j][i] = np.nanmin(dat)
                mean_values[j][i] = np.nanmean(dat)
                median_values[j][i] = np.nanmedian(dat)
                std_values[j][i] = np.nanstd(dat)
                diff_std_values[j][i] = np.std(np.diff(dat))
    win_features = list(chain(max_values, min_values, mean_values, median_values, std_values, diff_std_values))
    win_features = np.array(win_features).T
    suffixes = ['_max', '_min', '_mean', '_median', '_std', '_diff_std']
    new_feature_names = [f'{name}{suffix}' for suffix in suffixes for name in feature_names]
    result = pd.DataFrame(win_features, columns=new_feature_names)
    return (result, win_features)

def feature_empiric_score(dat):
    scores = np.zeros((len(dat), 8))
    for ii in range(len(dat)):
        HR = dat['HR'].iloc[ii]
        if pd.isna(HR):
            HR_score = np.nan
        elif (HR <= 40) | (HR >= 131):
            HR_score = 3
        elif 111 <= HR <= 130:
            HR_score = 2
        elif (41 <= HR <= 50) | (91 <= HR <= 110):
            HR_score = 1
        else:
            HR_score = 0
        scores[ii, 0] = HR_score
        Temp = dat['Temp'].iloc[ii]
        if pd.isna(Temp):
            Temp_score = np.nan
        elif Temp <= 35:
            Temp_score = 3
        elif Temp >= 39.1:
            Temp_score = 2
        elif (35.1 <= Temp <= 36.0) | (38.1 <= Temp <= 39.0):
            Temp_score = 1
        else:
            Temp_score = 0
        scores[ii, 1] = Temp_score
        Resp = dat['Resp'].iloc[ii]
        if pd.isna(Resp):
            Resp_score = np.nan
        elif (Resp < 8) | (Resp > 25):
            Resp_score = 3
        elif 21 <= Resp <= 24:
            Resp_score = 2
        elif 9 <= Resp <= 11:
            Resp_score = 1
        else:
            Resp_score = 0
        scores[ii, 2] = Resp_score
        Creatinine = dat['Creatinine'].iloc[ii]
        if pd.isna(Creatinine):
            Creatinine_score = np.nan
        elif Creatinine < 1.2:
            Creatinine_score = 0
        elif Creatinine < 2:
            Creatinine_score = 1
        elif Creatinine < 3.5:
            Creatinine_score = 2
        else:
            Creatinine_score = 3
        scores[ii, 3] = Creatinine_score
        MAP = dat['MAP'].iloc[ii]
        if pd.isna(MAP):
            MAP_score = np.nan
        elif MAP >= 70:
            MAP_score = 0
        else:
            MAP_score = 1
        scores[ii, 4] = MAP_score
        SBP = dat['SBP'].iloc[ii]
        Resp = dat['Resp'].iloc[ii]
        if pd.isna(SBP) or pd.isna(Resp):
            qsofa = np.nan
        elif (SBP <= 100) & (Resp >= 22):
            qsofa = 1
        else:
            qsofa = 0
        scores[ii, 5] = qsofa
        Platelets = dat['Platelets'].iloc[ii]
        if pd.isna(Platelets):
            Platelets_score = np.nan
        elif Platelets <= 50:
            Platelets_score = 3
        elif Platelets <= 100:
            Platelets_score = 2
        elif Platelets <= 150:
            Platelets_score = 1
        else:
            Platelets_score = 0
        scores[ii, 6] = Platelets_score
        Bilirubin = dat['Bilirubin_total'].iloc[ii]
        if pd.isna(Bilirubin):
            Bilirubin_score = np.nan
        elif Bilirubin < 1.2:
            Bilirubin_score = 0
        elif Bilirubin < 2:
            Bilirubin_score = 1
        elif Bilirubin < 6:
            Bilirubin_score = 2
        else:
            Bilirubin_score = 3
        scores[ii, 7] = Bilirubin_score
    score_names = ['HR_signature', 'Temp_signature', 'Resp_signature', 'Creatinine_signature', 'MAP_signature', 'qsofa_signature', 'Platelets_signature', 'Bilirubin_signature']
    result = pd.DataFrame(scores, columns=score_names)
    return (result, scores)

def qSOFA(df):
    qsofa = np.zeros(shape=df.shape[0])
    qsofa[df['Resp'].values >= 22] += 1
    qsofa[df['SBP'].values <= 100] += 1
    return qsofa

def SOFA(df):
    sofa = np.zeros(shape=df.shape[0])
    platelets = df['Platelets'].values
    sofa[platelets >= 150] += 0
    sofa[(100 <= platelets) & (platelets < 150)] += 1
    sofa[(50 <= platelets) & (platelets < 100)] += 2
    sofa[(20 <= platelets) & (platelets < 50)] += 3
    sofa[platelets < 20] += 4
    bilirubin = df['Bilirubin_total'].values
    sofa[bilirubin < 1.2] += 0
    sofa[(1.2 <= bilirubin) & (bilirubin <= 1.9)] += 1
    sofa[(1.9 < bilirubin) & (bilirubin <= 5.9)] += 2
    sofa[(5.9 < bilirubin) & (bilirubin <= 11.9)] += 3
    sofa[bilirubin > 11.9] += 4
    map = df['MAP'].values
    sofa[map >= 70] += 0
    sofa[map < 70] += 1
    creatinine = df['Creatinine'].values
    sofa[creatinine < 1.2] += 0
    sofa[(1.2 <= creatinine) & (creatinine <= 1.9)] += 1
    sofa[(1.9 < creatinine) & (creatinine <= 3.4)] += 2
    sofa[(3.4 < creatinine) & (creatinine <= 4.9)] += 3
    sofa[creatinine > 4.9] += 4
    return sofa

def SOFA_max_24(s):

    def find_24_hr_max(s, idx):
        start_idx = max(0, idx - 23)
        prev_24_hrs = s.iloc[start_idx:idx + 1]
        return np.nanmax(prev_24_hrs)
    sofa_24 = pd.Series(index=s.index, data=[find_24_hr_max(s, idx) for idx in s.index])
    return sofa_24

def SOFA_deterioration(s):

    def check_24hr_deterioration(s):
        prev_23_hrs = pd.concat([s.shift(i) for i in range(1, 24)], axis=1).values
        tfr_hr_min = np.full(s.shape[0], np.nan)
        for idx, row in enumerate(prev_23_hrs):
            valid_values = row[~np.isnan(row)]
            if valid_values.size > 0:
                tfr_hr_min[idx] = np.min(valid_values)
        return pd.Series(index=s.index, data=s.values - tfr_hr_min)
    sofa_det = check_24hr_deterioration(s)
    sofa_det[sofa_det < 0] = 0
    return sofa_det

def mews_score(df):
    mews = np.zeros(shape=df.shape[0])
    sbp = df['SBP'].values
    mews[sbp <= 70] += 3
    mews[(70 < sbp) & (sbp <= 80)] += 2
    mews[(80 < sbp) & (sbp <= 100)] += 1
    mews[sbp >= 200] += 2
    hr = df['HR'].values
    mews[hr < 40] += 2
    mews[(40 < hr) & (hr <= 50)] += 1
    mews[(100 < hr) & (hr <= 110)] += 1
    mews[(110 < hr) & (hr < 130)] += 2
    mews[hr >= 130] += 3
    resp = df['Resp'].values
    mews[resp < 9] += 2
    mews[(15 < resp) & (resp <= 20)] += 1
    mews[(20 < resp) & (resp < 30)] += 2
    mews[resp >= 30] += 3
    return mews

def septic_shock(df):
    shock = np.zeros(shape=df.shape[0])
    shock[df['MAP'].values < 65] += 1
    shock[df['Lactate'].values < 2] += 1
    return shock

def feature_authoritative_indicators(dat):
    dat['ShockIndex'] = dat['HR'].values / dat['SBP'].values
    dat['BUN/CR'] = dat['BUN'].values / dat['Creatinine'].values
    dat['SaO2/FiO2'] = np.divide(dat['SaO2'].values, dat['FiO2'].values, where=dat['FiO2'].values != 0, out=np.full_like(dat['SaO2'].values, np.nan))
    dat['SOFA'] = SOFA(dat[['Platelets', 'MAP', 'Creatinine', 'Bilirubin_total']])
    dat['SOFA_deterioration'] = SOFA_deterioration(dat['SOFA'])
    dat['SOFA_max_24hrs'] = SOFA_max_24(dat['SOFA'])
    dat['qSOFA'] = qSOFA(dat)
    dat['septic_shock'] = septic_shock(dat[['MAP', 'Lactate']])
    dat['MEWS'] = mews_score(dat)
    scores = np.zeros((len(dat), 9))
    scores[:, 0] = dat['ShockIndex'].values
    scores[:, 1] = dat['BUN/CR'].values
    scores[:, 2] = dat['SaO2/FiO2'].values
    scores[:, 3] = dat['SOFA'].values
    scores[:, 4] = dat['SOFA_deterioration'].values
    scores[:, 5] = dat['SOFA_max_24hrs'].values
    scores[:, 6] = dat['qSOFA'].values
    scores[:, 7] = dat['septic_shock'].values
    scores[:, 8] = dat['MEWS'].values
    score_names = ['ShockIndex', 'BUN/CR', 'SaO2/FiO2', 'SOFA', 'SOFA_deterioration', 'SOFA_max_24hrs', 'qSOFA', 'septic_shock', 'MEWS']
    result = pd.DataFrame(scores, columns=score_names)
    return (result, scores)

def feature_extraction(case, raw_columns=None, continuous=None, intermittent=None):
    use_raw_index = [name for name in raw_columns or _DEFAULT_SCHEMA['raw'] if name != 'FiO2']
    con_index = _DEFAULT_SCHEMA['continuous'] if continuous is None else continuous
    sep_index = _DEFAULT_SCHEMA['intermittent'] if intermittent is None else intermittent
    labels = np.array(case['SepsisLabel'])
    pid = case.drop(columns=['Bilirubin_direct', 'TroponinI', 'Fibrinogen', 'SepsisLabel', 'Age', 'Gender', 'Unit1', 'Unit2', 'ICULOS', 'HospAdmTime'], errors='ignore')
    temp_data = feature_informative_missingness(pid, con_index + sep_index)
    MissingFeature_containCol_A_ = pd.DataFrame(temp_data)
    MissingFeature_containCol_A_ = MissingFeature_containCol_A_.ffill()
    raw_data = MissingFeature_containCol_A_.loc[:, use_raw_index]
    missingFeatures = MissingFeature_containCol_A_.drop(columns=raw_data)
    feature_A = np.array(MissingFeature_containCol_A_)
    winFeature_containCol_B, feature_B = feature_slide_window(MissingFeature_containCol_A_, ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp'])
    empiric_containCol_C, feature_C = feature_empiric_score(MissingFeature_containCol_A_)
    authoritative_containCol_D, feature_D = feature_authoritative_indicators(MissingFeature_containCol_A_)
    features = pd.concat([raw_data, winFeature_containCol_B, empiric_containCol_C, authoritative_containCol_D, missingFeatures], axis=1)
    return (features, labels)
