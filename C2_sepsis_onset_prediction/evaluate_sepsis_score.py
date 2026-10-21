# Calculate the PhysioNet/CinC 2019 sepsis prediction scores.
import numpy as np, os, os.path, sys, warnings
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def evaluate_sepsis_score(label_directory, prediction_directory, name):
    label_header = 'SepsisLabel'
    prediction_header = 'PredictedLabel'
    probability_header = 'PredictedProbability'
    dt_early = -12
    dt_optimal = -6
    dt_late = 3
    max_u_tp = 1
    min_u_fn = -2
    u_fp = -0.05
    u_tn = 0
    label_files = []
    for f in os.listdir(label_directory):
        g = os.path.join(label_directory, f)
        if os.path.isfile(g) and (not f.lower().startswith('.')) and f.lower().endswith('npy'):
            label_files.append(g)
    label_files = sorted(label_files)
    prediction_files = []
    for f in os.listdir(prediction_directory):
        g = os.path.join(prediction_directory, f)
        if os.path.isfile(g) and (not f.lower().startswith('.')) and f.lower().endswith('npy'):
            prediction_files.append(g)
    prediction_files = sorted(prediction_files)
    if len(label_files) != len(prediction_files):
        raise Exception('Numbers of label and prediction files must be the same.')
    num_files = len(label_files)
    cohort_labels = []
    cohort_predictions = []
    cohort_probabilities = []
    for k in range(num_files):
        labels = load_column(label_files[k], label_header, '|')
        predictions = load_column(prediction_files[k], prediction_header, '|')
        probabilities = load_column(prediction_files[k], probability_header, '|')
        if not (len(labels) == len(predictions) and len(predictions) == len(probabilities)):
            raise Exception('Numbers of labels and predictions for a file must be the same.')
        num_rows = len(labels)
        for i in range(num_rows):
            if labels[i] not in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')
            if predictions[i] not in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')
            if not 0 <= probabilities[i] <= 1:
                warnings.warn('Probabilities do not satisfy 0 <= probability <= 1.')
        if 0 < np.sum(predictions) < num_rows:
            min_probability_positive = np.min(probabilities[predictions == 1])
            max_probability_negative = np.max(probabilities[predictions == 0])
            if min_probability_positive <= max_probability_negative:
                warnings.warn('Predictions are inconsistent with probabilities, i.e., a positive prediction has a lower (or equal) probability than a negative prediction.')
        cohort_labels.append(labels)
        cohort_predictions.append(predictions)
        cohort_probabilities.append(probabilities)
    labels = np.concatenate(cohort_labels)
    predictions = np.concatenate(cohort_predictions)
    probabilities = np.concatenate(cohort_probabilities)
    auroc, auprc = compute_auc(labels, probabilities)
    accuracy, f_measure = compute_accuracy_f_measure(labels, predictions)
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    data = dict(fpr=fpr, tpr=tpr, precision=precision, recall=recall, y_true=labels, y_pred=predictions, y_prob=probabilities)
    np.savez(name + '.npz', **data)
    observed_utilities = np.zeros(num_files)
    best_utilities = np.zeros(num_files)
    worst_utilities = np.zeros(num_files)
    inaction_utilities = np.zeros(num_files)
    for k in range(num_files):
        labels = cohort_labels[k]
        num_rows = len(labels)
        observed_predictions = cohort_predictions[k]
        best_predictions = np.zeros(num_rows)
        worst_predictions = np.zeros(num_rows)
        inaction_predictions = np.zeros(num_rows)
        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            best_predictions[max(0, t_sepsis + dt_early):min(t_sepsis + dt_late + 1, num_rows)] = 1
        worst_predictions = 1 - best_predictions
        observed_utilities[k] = compute_prediction_utility(labels, observed_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        best_utilities[k] = compute_prediction_utility(labels, best_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        worst_utilities[k] = compute_prediction_utility(labels, worst_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        inaction_utilities[k] = compute_prediction_utility(labels, inaction_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
    unnormalized_observed_utility = np.sum(observed_utilities)
    unnormalized_best_utility = np.sum(best_utilities)
    unnormalized_worst_utility = np.sum(worst_utilities)
    unnormalized_inaction_utility = np.sum(inaction_utilities)
    normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility)
    return (auroc, auprc, accuracy, f_measure, normalized_observed_utility, sensitivity, specificity)

def load_column(filename, header, delimiter):
    column = []
    with open(filename, 'r') as f:
        for i, l in enumerate(f):
            arrs = l.strip().split(delimiter)
            if i == 0:
                try:
                    j = arrs.index(header)
                except:
                    raise Exception('{} must contain column with header {} containing numerical entries.'.format(filename, header))
            elif len(arrs[j]):
                column.append(float(arrs[j]))
    return np.array(column)

def compute_auc(labels, predictions, check_errors=True):
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')
        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')
        for prediction in predictions:
            if not 0 <= prediction <= 1:
                warnings.warn('Predictions do not satisfy 0 <= prediction <= 1.')
    thresholds = np.unique(predictions)[::-1]
    if thresholds[0] != 1:
        thresholds = np.insert(thresholds, 0, 1)
    if thresholds[-1] == 0:
        thresholds = thresholds[:-1]
    n = len(labels)
    m = len(thresholds)
    tp = np.zeros(m)
    fp = np.zeros(m)
    fn = np.zeros(m)
    tn = np.zeros(m)
    idx = np.argsort(predictions)[::-1]
    i = 0
    for j in range(m):
        if j == 0:
            tp[j] = 0
            fp[j] = 0
            fn[j] = np.sum(labels)
            tn[j] = n - fn[j]
        else:
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]
        while i < n and predictions[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1
    tpr = np.zeros(m)
    tnr = np.zeros(m)
    ppv = np.zeros(m)
    npv = np.zeros(m)
    for j in range(m):
        if tp[j] + fn[j]:
            tpr[j] = tp[j] / (tp[j] + fn[j])
        else:
            tpr[j] = 1
        if fp[j] + tn[j]:
            tnr[j] = tn[j] / (fp[j] + tn[j])
        else:
            tnr[j] = 1
        if tp[j] + fp[j]:
            ppv[j] = tp[j] / (tp[j] + fp[j])
        else:
            ppv[j] = 1
        if fn[j] + tn[j]:
            npv[j] = tn[j] / (fn[j] + tn[j])
        else:
            npv[j] = 1
    auroc = 0
    auprc = 0
    for j in range(m - 1):
        auroc += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
        auprc += (tpr[j + 1] - tpr[j]) * ppv[j + 1]
    return (auroc, auprc)

def compute_accuracy_f_measure(labels, predictions, check_errors=True):
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')
        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')
        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')
    n = len(labels)
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(n):
        if labels[i] and predictions[i]:
            tp += 1
        elif not labels[i] and predictions[i]:
            fp += 1
        elif labels[i] and (not predictions[i]):
            fn += 1
        elif not labels[i] and (not predictions[i]):
            tn += 1
    if tp + fp + fn + tn:
        accuracy = float(tp + tn) / float(tp + fp + fn + tn)
    else:
        accuracy = 1.0
    if 2 * tp + fp + fn:
        f_measure = float(2 * tp) / float(2 * tp + fp + fn)
    else:
        f_measure = 1.0
    return (accuracy, f_measure)

def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0, check_errors=True):
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')
        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')
        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')
        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')
        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')
    n = len(labels)
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            elif is_septic and (not predictions[t]):
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and (not predictions[t]):
                u[t] = u_tn
    return np.sum(u)
if __name__ == '__main__':
    auroc, auprc, accuracy, f_measure, utility = evaluate_sepsis_score(sys.argv[1], sys.argv[2])
    output_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility\n{}|{}|{}|{}|{}'.format(auroc, auprc, accuracy, f_measure, utility)
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'w') as f:
            f.write(output_string)
    else:
        print(output_string)
