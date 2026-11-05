"""Lesson 7: metrics and decision thresholds on explicitly constructed data.

NumPy only. No classifier is trained: input scores are fixed teaching examples.
The weights demo solves a constant-probability model analytically.
Confusion matrices: rows=true, columns=predicted, class order [0, 1].
"""

import argparse
import math

import numpy as np


def binary_vector(values):
    values = np.asarray(values)
    if values.ndim != 1 or not values.size or not np.all((values == 0) | (values == 1)):
        raise ValueError("Use a nonempty 1-D binary vector with labels 0 and 1.")
    return values.astype(int)


def binary_confusion(y_true, y_pred):
    y_true, y_pred = binary_vector(y_true), binary_vector(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Targets and predictions must have identical shapes.")
    return np.bincount(2*y_true+y_pred, minlength=4).reshape(2, 2)


def count_matrix(cm):
    cm = np.asarray(cm, dtype=float)
    if (cm.ndim != 2 or cm.shape[0] != cm.shape[1] or not cm.size
            or not np.all(np.isfinite(cm)) or np.any(cm < 0)
            or np.any(cm != np.floor(cm)) or cm.sum() == 0):
        raise ValueError("Use a nonempty square matrix of nonnegative integer counts.")
    return cm


def ratio(numerator, denominator, zero_division=math.nan):
    """Undefined scalar ratios stay NaN by default; allow explicit 0 or 1."""
    return float(numerator/denominator) if denominator else float(zero_division)


def binary_metrics(cm, zero_division=math.nan):
    cm = count_matrix(cm)
    if cm.shape != (2, 2):
        raise ValueError("Binary confusion matrix must have shape (2,2).")
    tn, fp, fn, tp = cm.ravel()
    recall = ratio(tp, tp+fn, zero_division)
    specificity = ratio(tn, tn+fp, zero_division)
    # Do not silently ignore an absent true class when averaging recalls.
    balanced = (recall+specificity)/2 if (tp+fn > 0 and tn+fp > 0) else math.nan
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "support_positive": int(tp+fn), "predicted_positive": int(tp+fp),
            "accuracy": float((tp+tn)/cm.sum()),
            "precision": ratio(tp, tp+fp, zero_division), "recall": recall,
            "specificity": specificity, "fpr": ratio(fp, fp+tn, zero_division),
            "f1": ratio(2*tp, 2*tp+fp+fn, zero_division),
            "balanced_accuracy": balanced}


def score_inputs(y_true, scores):
    y_true = binary_vector(y_true)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != y_true.shape or not np.all(np.isfinite(scores)):
        raise ValueError("Use matching 1-D labels and finite scores.")
    return y_true, scores


def threshold_rows(y_true, scores, candidates=(.7, .5, .4), fp_cost=1., fn_cost=5.):
    y_true, scores = score_inputs(y_true, scores)
    if len(candidates) == 0 or not np.all(np.isfinite(candidates)):
        raise ValueError("Provide finite candidate thresholds.")
    if min(fp_cost, fn_cost) < 0 or not np.all(np.isfinite([fp_cost, fn_cost])):
        raise ValueError("Use finite nonnegative costs.")
    rows = []
    for threshold in candidates:
        row = binary_metrics(binary_confusion(y_true, scores >= threshold))
        row.update(threshold=float(threshold), cost=fp_cost*row['fp']+fn_cost*row['fn'])
        rows.append(row)
    return rows


def choose_threshold(y_val, scores_val, candidates=(.7, .5, .4), fp_cost=1., fn_cost=5.):
    """No test inputs. Cost ties choose the larger threshold, explicitly."""
    rows = threshold_rows(y_val, scores_val, candidates, fp_cost, fn_cost)
    return min(rows, key=lambda row: (row['cost'], -row['threshold'])), rows


def validation_scores():
    return (np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
            np.array([.95, .9, .8, .7, .6, .6, .45, .4, .3, .2, .1, .05]))


def final_test_scores():
    return (np.array([0, 1, 0, 0, 1, 0, 0, 0]),
            np.array([.85, .75, .65, .5, .35, .25, .15, .05]))


def ranking_example():
    return np.array([1, 0, 1, 0]), np.array([.9, .8, .8, .1])


def ranking_curves(y_true, scores):
    """Small O(n*k) teaching implementation; ties enter together.

    Both true classes are required here. Thresholds descend, so recall rises.
    PR origin precision=1 is a plotting convention, not measured precision.
    """
    y_true, scores = score_inputs(y_true, scores)
    if len(np.unique(y_true)) != 2:
        raise ValueError("This curve demo requires both true classes.")
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    precision, recall, fpr = [1.], [0.], [0.]
    for threshold in thresholds[1:]:
        row = binary_metrics(binary_confusion(y_true, scores >= threshold))
        precision.append(row['precision']); recall.append(row['recall']); fpr.append(row['fpr'])
    precision, recall, fpr = map(np.asarray, (precision, recall, fpr))
    ap = float(np.sum(np.diff(recall)*precision[1:]))
    return {"thresholds": thresholds, "precision": precision, "recall": recall,
            "fpr": fpr, "ap": ap, "roc_auc": float(np.trapezoid(recall, fpr)),
            "pr_trapezoid": float(np.trapezoid(precision, recall))}


def prevalence_precision(prevalence, tpr=.8, fpr=.02):
    prevalence = np.asarray(prevalence, dtype=float)
    if (np.any((prevalence <= 0) | (prevalence >= 1)) or not np.all(np.isfinite(prevalence))
            or not 0 <= tpr <= 1 or not 0 <= fpr <= 1):
        raise ValueError("Use 0<prevalence<1 and finite rates between 0 and 1.")
    denominator = tpr*prevalence+fpr*(1-prevalence)
    return np.divide(tpr*prevalence, denominator, out=np.full_like(prevalence, np.nan),
                     where=denominator > 0)


def multiclass_counts():
    # Class order [billing, technical, account]; each sample has exactly one label.
    return np.array([[87, 2, 1], [4, 3, 1], [1, 1, 0]])


def multiclass_report(cm, zero_division=0.):
    """Fixed full class list. Explicit zero-division=0 for this report."""
    cm = count_matrix(cm)
    support, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    precision = np.array([ratio(a, b, zero_division) for a, b in zip(tp, predicted)])
    recall = np.array([ratio(a, b, zero_division) for a, b in zip(tp, support)])
    f1 = np.array([ratio(2*a, b+c, zero_division) for a, b, c in zip(tp, support, predicted)])
    accuracy = float(tp.sum()/cm.sum())
    return {"support": support.astype(int), "precision": precision, "recall": recall, "f1": f1,
            "macro_f1": float(f1.mean()), "weighted_f1": float(support@f1/cm.sum()),
            "micro_f1": accuracy, "accuracy": accuracy,
            "macro_recall": float(recall.mean())}


def weighted_bce(y_true, logits, weights):
    y_true, logits = score_inputs(y_true, logits)
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != y_true.shape or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Use matching finite nonnegative weights with positive total.")
    losses = np.logaddexp(0, np.where(y_true == 1, -logits, logits))
    return float(weights@losses/weights.sum())


def constant_fit(y_true, positive_weight=1.):
    """Exact optimum of weighted BCE for one constant p; no features or GD."""
    y_true = binary_vector(y_true)
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("Use a finite positive class weight.")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Use both classes to get a finite optimal constant logit.")
    weights = np.where(y_true == 1, positive_weight, 1.)
    p = float(weights@y_true/weights.sum())
    z = math.log(p/(1-p))
    return p, weighted_bce(y_true, np.full(len(y_true), z), weights)


def print_metrics(row):
    keys = ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'balanced_accuracy']
    print(' '.join(f'{key}={row[key]:.6f}' for key in keys))


def hand_demo():
    cm = np.array([[950, 30], [5, 15]])
    print('rows=true, columns=predicted, class order [0,1]:\n', cm)
    print_metrics(binary_metrics(cm))
    print('All-negative baseline:')
    print_metrics(binary_metrics(np.array([[980, 0], [20, 0]])))
    print('Cost FP+10*FN: model=80; all-negative=200 (constructed example).')


def threshold_demo():
    best, rows = choose_threshold(*validation_scores())
    for row in rows:
        print(f"validation threshold={row['threshold']:g}; TP={row['tp']}; FP={row['fp']}; "
              f"FN={row['fn']}; cost={row['cost']:g}; F1={row['f1']:.6f}; accuracy={row['accuracy']:.6f}")
    print(f"Frozen validation choice: threshold={best['threshold']:g}")
    # Test is accessed only after the threshold is frozen. No candidate test sweep.
    y_test, scores_test = final_test_scores()
    row = threshold_rows(y_test, scores_test, (best['threshold'],))[0]
    print('Final test matrix:\n', binary_confusion(y_test, scores_test >= best['threshold']))
    print_metrics(row)
    print(f"Final test cost={row['cost']:g}; all-negative baseline cost=10; accuracy=0.75")


def curves_demo():
    curves = ranking_curves(*ranking_example())
    for key, values in curves.items(): print(key, values)


def prevalence_demo():
    for prevalence in [.1, .01]:
        pos, neg = 10000*prevalence, 10000*(1-prevalence)
        print(f'Expected counts n=10000, prevalence={prevalence:g}: TP={.8*pos:g}, '
              f'FN={.2*pos:g}, FP={.02*neg:g}, TN={.98*neg:g}; '
              f'precision={float(prevalence_precision(prevalence)):.6f}')


def multiclass_demo():
    cm = multiclass_counts(); report = multiclass_report(cm)
    print('rows=true, columns=predicted; [billing, technical, account]\n', cm)
    for key, values in report.items(): print(key, values)


def weights_demo():
    y = np.r_[np.zeros(98, dtype=int), np.ones(2, dtype=int)]
    for weight in (1., 49.):
        p, loss = constant_fit(y, weight)
        print(f'positive_weight={weight:g}; optimal constant p={p:.6f}; weighted mean BCE={loss:.6f}')
    print('Weighting changes the objective; p=0.5 is not the natural 2% class frequency.')


def undefined_demo():
    for name, cm in [('no predicted positives', [[3, 0], [2, 0]]),
                     ('only true negatives', [[5, 0], [0, 0]])]:
        print(name, 'default NaN:'); print_metrics(binary_metrics(cm))
        print('explicit zero_division=0:'); print_metrics(binary_metrics(cm, zero_division=0))


def main():
    demos = {'hand': hand_demo, 'threshold': threshold_demo, 'curves': curves_demo,
             'prevalence': prevalence_demo, 'multiclass': multiclass_demo,
             'weights': weights_demo, 'undefined': undefined_demo}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=[*demos, 'all'], default='all')
    args = parser.parse_args()
    for name, demo in demos.items():
        if args.mode in (name, 'all'):
            print(f'\n--- {name} ---'); demo()


if __name__ == '__main__':
    main()
