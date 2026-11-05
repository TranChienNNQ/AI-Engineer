"""Lesson 5: split roles, model selection, preprocessing, groups and time.

All datasets are synthetic and deliberately small. NumPy only.
Run python3 lab.py --mode all. A stable tie rule makes hand calculations repeatable.
"""

import argparse

import numpy as np


SEED = 2026


def training_data():
    return np.arange(6, dtype=float), np.array([10., 12., 11., 14., 13., 15.])


def validation_data():
    x = np.array([0.5, 2.5, 4.5])
    return x, 10 + x


def test_data():
    x = np.array([1.5, 3.5])
    return x, 10 + x


def knn_predict(train_x, train_y, query_x, k):
    """Average k nearest training targets in 1-D; ties keep training-row order."""
    if int(k) != k or not 1 <= k <= len(train_x):
        raise ValueError("Use integer k between 1 and the number of training rows.")
    if train_x.ndim != 1 or train_y.shape != train_x.shape or query_x.ndim != 1:
        raise ValueError("Use matching 1-D training arrays and 1-D queries.")
    predictions=[]
    for x in query_x:
        nearest=np.argsort(np.abs(train_x - x), kind="stable")[:int(k)]
        predictions.append(float(np.mean(train_y[nearest])))
    return np.array(predictions)


def mse(y, pred):
    if y.ndim != 1 or y.shape != pred.shape or not len(y):
        raise ValueError("Use matching nonempty 1-D target and prediction arrays.")
    return float(np.mean((y - pred) ** 2))


def choose_k(train_x, train_y, val_x, val_y, candidates=(1, 3, 5)):
    """Choose using validation only. This function has no test-data input."""
    rows=[]
    for k in candidates:
        train_loss=mse(train_y, knn_predict(train_x, train_y, train_x, k))
        val_loss=mse(val_y, knn_predict(train_x, train_y, val_x, k))
        rows.append((k, train_loss, val_loss))
    # In a tie, the earlier candidate wins; this rule is fixed before evaluation.
    selected=min(rows, key=lambda row: row[2])[0]
    return selected, rows


def fit_scaler(train_x):
    """Fit mean/std from the supplied training data only, ddof=0."""
    mean=float(np.mean(train_x))
    scale=float(np.std(train_x, ddof=0))
    return mean, scale if scale > 0 else 1.0


def transform(x, mean, scale):
    return (x - mean) / scale


def grouped_data():
    # Arbitrary balanced labels per user; the memorizer does NOT use id parity.
    groups=np.repeat(np.arange(12), 6)
    labels=np.repeat(np.tile([0, 1], 6), 6)
    return groups, labels


def group_partitions(mode):
    groups,_=grouped_data()
    if mode == "row":
        position=np.tile(np.arange(6), 12)
        return np.flatnonzero(position < 4), np.flatnonzero(position == 4), np.flatnonzero(position == 5)
    if mode == "group":
        return np.flatnonzero(groups < 8), np.flatnonzero((groups >= 8) & (groups < 10)), np.flatnonzero(groups >= 10)
    raise ValueError("Use row or group.")


def fit_memorizer(groups, labels):
    """Remember the majority label of seen groups; tie rule and fallback use 0."""
    mapping={int(g): int(np.mean(labels[groups == g]) > 0.5) for g in np.unique(groups)}
    fallback=int(np.mean(labels) > 0.5)
    return mapping, fallback


def memorizer_predict(groups, mapping, fallback):
    return np.array([mapping.get(int(g), fallback) for g in groups])


def group_results():
    groups,labels=grouped_data()
    results=[]
    for mode in ("row", "group"):
        train,val,test=group_partitions(mode)
        mapping,fallback=fit_memorizer(groups[train],labels[train])
        accuracy=float(np.mean(memorizer_predict(groups[test],mapping,fallback)==labels[test]))
        overlap=len(set(groups[train]) & set(groups[test]))
        results.append((mode,len(train),len(val),len(test),overlap,accuracy))
    return results


def temporal_partitions():
    """Fit before t=9; labels available at decision+2 must be strictly before cutoff."""
    decision=np.arange(1,15)
    label_available=decision + 2
    raw_train=decision <= 8
    val=(decision >= 9) & (decision <= 10)
    test=decision >= 13
    train=raw_train & (label_available < 9)
    purged=raw_train & ~train
    waiting=(decision == 11) | (decision == 12)
    return decision,label_available,train,val,test,purged,waiting


def cv_folds(n=6, seed=SEED):
    order=np.random.default_rng(seed).permutation(n)
    return list(np.array_split(order, 3))


def cv_results(seed=SEED):
    """Three folds on development data only; fit scaler afresh inside each fold."""
    x,y=training_data()
    folds=cv_folds(len(x),seed)
    results=[]
    for k in (1,3):
        losses=[]
        for fold in range(3):
            val=folds[fold]
            train=np.concatenate([folds[j] for j in range(3) if j != fold])
            mean,scale=fit_scaler(x[train])
            pred=knn_predict(transform(x[train],mean,scale),y[train],transform(x[val],mean,scale),k)
            losses.append(mse(y[val],pred))
        results.append((k,losses,float(np.mean(losses))))
    return results


def split_demo(seed):
    ids=np.random.default_rng(seed).permutation(20)
    for name,part in [("train",ids[:12]),("validation",ids[12:16]),("test",ids[16:])]:
        print(f"{name}: n={len(part)}, row_ids={part}")
    print("Disjoint row IDs do not prove independent users/documents/time windows.")


def selection_demo():
    train_x,train_y=training_data()
    val_x,val_y=validation_data()
    selected,rows=choose_k(train_x,train_y,val_x,val_y)
    print("k   train_MSE   validation_MSE")
    for k,tr,va in rows:
        print(f"{k}   {tr:.6f}    {va:.6f}")
    print(f"Selected and frozen k={selected}")
    # Access the test examples only after choosing the candidate.
    tx,ty=test_data()
    pred=knn_predict(train_x,train_y,tx,selected)
    print("Selected-model test predictions:",pred)
    print(f"Selected-model test MSE: {mse(ty,pred):.6f}")


def preprocess_demo():
    train=np.array([0.,2.]);test=np.array([10.])
    mean,scale=fit_scaler(train)
    leaked_mean,leaked_scale=fit_scaler(np.r_[train,test])
    print(f"Train-only mean={mean:.6f}, scale={scale:.6f}; test z={transform(test,mean,scale)[0]:.6f}")
    print(f"Leaked mean={leaked_mean:.6f}, scale={leaked_scale:.6f}; test z={transform(test,leaked_mean,leaked_scale)[0]:.6f}")
    print("This demonstrates contaminated fitted statistics, not a guaranteed score improvement.")


def groups_demo():
    for mode,tr,va,te,overlap,acc in group_results():
        print(f"{mode}: train/val/test={tr}/{va}/{te}; shared train-test groups={overlap}; test accuracy={acc:.6f}")
    print("The estimands differ: seen users versus new users; this is a constructed example.")


def time_demo():
    t,available,train,val,test,purged,waiting=temporal_partitions()
    print("Train decisions allowed before t=9:",t[train])
    print("Purged training decisions:",t[purged],"; labels available:",available[purged])
    print("Validation decisions:",t[val],"; labels available:",available[val])
    print("Wait for validation labels at t=11,12:",t[waiting])
    print("Frozen-test decisions:",t[test],"; labels available:",available[test])
    print("Policy: training-label availability < first validation decision;")
    print("validation-label availability < first test decision.")


def cv_demo(seed):
    for k,losses,average in cv_results(seed):
        print(f"k={k}; fold validation MSE={np.array(losses)}; mean={average:.6f}")
    print("CV chooses a setting on development data. No final-test score is calculated here.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["all","split","select","preprocess","groups","time","cv"],default="all")
    parser.add_argument("--seed",type=int,default=SEED)
    args=parser.parse_args()
    if args.seed < 0: parser.error("Use seed >= 0.")
    np.set_printoptions(precision=6,suppress=True)
    demos={"split":lambda:split_demo(args.seed),"select":selection_demo,"preprocess":preprocess_demo,
           "groups":groups_demo,"time":time_demo,"cv":lambda:cv_demo(args.seed)}
    for name,demo in demos.items():
        if args.mode in ("all",name):
            print(f"\n--- {name} ---")
            demo()


if __name__ == "__main__":
    main()
