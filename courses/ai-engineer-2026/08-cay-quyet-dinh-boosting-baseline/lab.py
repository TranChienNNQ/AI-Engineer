"""Lesson 8: small numeric trees and squared-error gradient boosting.

Teaching implementation: binary classification or regression, finite numeric X,
no weights, missing values, pruning, categorical splits or production tuning.
Train/validation select the model; only the selected model and fixed baselines
are scored on test. All data are constructed, not production measurements.
"""

import argparse
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def training_arrays(X, y, task):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if (X.ndim != 2 or y.ndim != 1 or X.shape[0] != len(y)
            or len(y) == 0 or X.shape[1] == 0):
        raise ValueError('Use nonempty X=(n,d) and matching y=(n,).')
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError('This lab requires finite numeric data; handle missing values explicitly.')
    if task not in ('regression', 'binary'):
        raise ValueError('Task must be regression or binary.')
    if task == 'binary' and not np.all((y == 0) | (y == 1)):
        raise ValueError('Binary labels must be 0 or 1.')
    return X, y


def impurity(y, task):
    if task == 'binary':
        p = float(np.mean(y))
        return 2*p*(1-p)
    return float(np.mean((y-np.mean(y))**2))


def split_candidates(X, y, task='regression', min_leaf=1):
    """Weighted child impurity for each midpoint of adjacent distinct values.

    Sorted by feature index then threshold. Ties within EPS keep the first.
    This exhaustive, repeated-mask implementation favors readability over speed.
    """
    X, y = training_arrays(X, y, task)
    if not isinstance(min_leaf, (int, np.integer)) or min_leaf < 1:
        raise ValueError('min_leaf must be a positive integer.')
    parent = impurity(y, task)
    rows = []
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        for lower, upper in zip(values[:-1], values[1:]):
            threshold = float(lower/2+upper/2)
            left = X[:, feature] <= threshold
            nl, nr = int(left.sum()), int((~left).sum())
            if min(nl, nr) < min_leaf:
                continue
            score = (nl*impurity(y[left], task)+nr*impurity(y[~left], task))/len(y)
            rows.append({'feature': feature, 'threshold': threshold,
                         'left_n': nl, 'right_n': nr, 'score': score, 'gain': parent-score})
    return rows


@dataclass
class Node:
    value: float
    n: int
    impurity: float
    feature: int | None = None
    threshold: float | None = None
    left: 'Node | None' = None
    right: 'Node | None' = None


class SmallTree:
    def __init__(self, task='regression', max_depth=2, min_leaf=1):
        if task not in ('regression', 'binary'):
            raise ValueError('Task must be regression or binary.')
        if not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
            raise ValueError('max_depth must be a nonnegative integer.')
        if not isinstance(min_leaf, (int, np.integer)) or min_leaf < 1:
            raise ValueError('min_leaf must be a positive integer.')
        self.task, self.max_depth, self.min_leaf = task, max_depth, min_leaf

    def fit(self, X, y):
        X, y = training_arrays(X, y, self.task)
        self.n_features = X.shape[1]
        self.root = self._grow(X, y, 0)
        return self

    def _grow(self, X, y, depth):
        node = Node(float(y.mean()), len(y), impurity(y, self.task))
        if depth >= self.max_depth or len(y) < 2*self.min_leaf or node.impurity <= EPS:
            return node
        best = None
        for row in split_candidates(X, y, self.task, self.min_leaf):
            if best is None or row['score'] < best['score']-EPS:
                best = row
        # Require a strictly positive immediate gain; this intentionally stops on pure XOR.
        if best is None or best['gain'] <= EPS:
            return node
        node.feature, node.threshold = best['feature'], best['threshold']
        left = X[:, node.feature] <= node.threshold
        node.left = self._grow(X[left], y[left], depth+1)
        node.right = self._grow(X[~left], y[~left], depth+1)
        return node

    def predict(self, X):
        """Returns numeric predictions for regression; leaf P(Y=1) for binary."""
        if not hasattr(self, 'root'):
            raise ValueError('Fit the tree before predicting.')
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features or not np.all(np.isfinite(X)):
            raise ValueError('Use finite X=(n,d) with the fitted feature order.')
        def one(row):
            node = self.root
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            return node.value
        return np.array([one(row) for row in X])


def mse(y, predictions):
    y, predictions = np.asarray(y), np.asarray(predictions)
    if y.ndim != 1 or y.shape != predictions.shape or y.size == 0:
        raise ValueError('Use matching nonempty one-dimensional arrays.')
    return float(np.mean((y-predictions)**2))


class SquaredBoost:
    """F0=mean(y); regress y-F onto small trees; F += learning_rate*tree.

    This is squared-error boosting, not a general loss or classifier implementation.
    """
    def __init__(self, rounds=20, learning_rate=.1, max_depth=1, min_leaf=2):
        if not isinstance(rounds, (int, np.integer)) or rounds < 0:
            raise ValueError('rounds must be a nonnegative integer.')
        if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError('This lab restricts learning_rate to (0,1].')
        SmallTree('regression', max_depth, min_leaf)  # Validate shared tree settings.
        self.rounds, self.learning_rate = rounds, learning_rate
        self.max_depth, self.min_leaf = max_depth, min_leaf

    def fit(self, X, y):
        X, y = training_arrays(X, y, 'regression')
        self.initial, self.n_features = float(y.mean()), X.shape[1]
        predictions = np.full(len(y), self.initial)
        self.trees, self.train_history = [], [mse(y, predictions)]
        for _ in range(self.rounds):
            residual = y-predictions
            tree = SmallTree('regression', self.max_depth, self.min_leaf).fit(X, residual)
            predictions += self.learning_rate*tree.predict(X)
            self.trees.append(tree)
            self.train_history.append(mse(y, predictions))
        self.train_history = np.array(self.train_history)
        return self

    def staged_predict(self, X):
        if not hasattr(self, 'trees'):
            raise ValueError('Fit boosting before predicting.')
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features or not np.all(np.isfinite(X)):
            raise ValueError('Use finite X=(n,d) with the fitted feature order.')
        predictions = np.full(len(X), self.initial)
        yield predictions.copy()  # Stage zero is the train-mean baseline.
        for tree in self.trees:
            predictions += self.learning_rate*tree.predict(X)
            yield predictions.copy()

    def predict(self, X):
        predictions = None
        for predictions in self.staged_predict(X):
            pass
        return predictions


def hand_classification():
    return np.arange(1, 7, dtype=float)[:, None], np.array([0., 0., 1., 0., 1., 1.])


def hand_regression():
    return np.arange(1, 5, dtype=float)[:, None], np.array([1., 1., 3., 5.])


def region_data():
    X = np.array([[load, error] for error in (0., 2.) for load in (1., 2., 3., 4.)])
    y = ((X[:, 0] > 2.5) & (X[:, 1] > 1)).astype(float)
    return X, y


def signal(x):
    """Known synthetic signal; not a feature available from future labels."""
    return 20+2*x+8*(x >= 5)


def development_data():
    tx = np.linspace(0, 10, 32)
    vx = np.linspace(.15, 9.85, 21)
    train = (tx[:, None], signal(tx)+np.random.default_rng(2026).normal(0, 2.2, len(tx)))
    val = (vx[:, None], signal(vx)+np.random.default_rng(2027).normal(0, .8, len(vx)))
    return train, val


def final_test_data():
    x = np.array([.35, 1.7, 3.3, 4.9, 5.1, 6.6, 8.2, 9.6])
    noise = np.array([.3, -.4, .2, -.2, .4, -.3, .1, -.1])
    return x[:, None], signal(x)+noise


def fit_mean(X, y):
    X, y = training_arrays(X, y, 'regression')
    return {'kind': 'mean', 'value': float(y.mean()), 'features': X.shape[1]}


def fit_linear(X, y):
    X, y = training_arrays(X, y, 'regression')
    A = np.column_stack([np.ones(len(y)), X])
    return {'kind': 'linear', 'coef': np.linalg.lstsq(A, y, rcond=None)[0], 'features': X.shape[1]}


def model_predict(model, X):
    if isinstance(model, dict):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != model['features'] or not np.all(np.isfinite(X)):
            raise ValueError('Use finite X with fitted feature order.')
        if model['kind'] == 'mean':
            return np.full(len(X), model['value'])
        return np.column_stack([np.ones(len(X)), X])@model['coef']
    return model.predict(X)


def select_model(train, val):
    """Fixed candidate grid, choose by validation MSE. Does not access test."""
    X, y = train; vx, vy = val
    models = [('mean', fit_mean(X, y)), ('linear', fit_linear(X, y))]
    for depth in (1, 2, 4):
        models.append((f'tree depth={depth}', SmallTree('regression', depth, 2).fit(X, y)))
    for rounds in (1, 5, 20, 80, 200):
        models.append((f'boost M={rounds}', SquaredBoost(rounds, .1, 1, 2).fit(X, y)))
    rows = [{'name': name, 'model': model, 'train_mse': mse(y, model_predict(model, X)),
             'val_mse': mse(vy, model_predict(model, vx))} for name, model in models]
    # Within EPS, the first candidate in the declared order wins.
    best = rows[0]
    for row in rows[1:]:
        if row['val_mse'] < best['val_mse']-EPS:
            best = row
    return best, rows


def classification_demo():
    X, y = hand_classification()
    print('Root Gini:', impurity(y, 'binary'))
    for row in split_candidates(X, y, 'binary'):
        print(f"threshold={row['threshold']:g}; child_Gini={row['score']:.6f}; gain={row['gain']:.6f}")
    tree = SmallTree('binary', 1, 1).fit(X, y)
    p = tree.predict(X); labels = (p >= .5).astype(int)
    print('Stump leaf probabilities:', p, '; labels at >=0.5:', labels)
    print('TRAIN accuracy:', np.mean(labels == y), '; not held-out evaluation.')
    print('Train-prior baseline p:', y.mean(), '; majority tie convention here selects class 0.')


def regression_demo():
    X, y = hand_regression()
    print('Mean baseline:', y.mean(), '; MSE:', impurity(y, 'regression'))
    for row in split_candidates(X, y):
        print(f"threshold={row['threshold']:g}; SSE={len(y)*row['score']:.6f}; MSE={row['score']:.6f}")
    tree = SmallTree('regression', 1, 1).fit(X, y)
    print('Stump predictions:', tree.predict(X), '; train MSE:', mse(y, tree.predict(X)))
    print('At x=0 and x=10:', tree.predict([[0.], [10.]]), '(piecewise constant extrapolation)')


def boosting_demo():
    X, y = hand_regression()
    model = SquaredBoost(2, .5, 1, 1).fit(X, y)
    previous = None
    for stage, predictions in enumerate(model.staged_predict(X)):
        if stage:
            print('Residual fit target:', y-previous)
            print('New stump threshold:', model.trees[stage-1].root.threshold)
        print(f'Stage {stage}: prediction={predictions}; MSE={mse(y, predictions):.6f}')
        previous = predictions


def comparison_demo():
    train, val = development_data()
    best, rows = select_model(train, val)
    print('Candidate                  Train MSE      Validation MSE')
    for row in rows:
        print(f"{row['name']:<27}{row['train_mse']:<15.6f}{row['val_mse']:.6f}")
    print('Frozen selection:', best['name'], '; no train+validation refit.')
    X, y = final_test_data()  # Only after selection; score selected plus two fixed baselines.
    chosen = [best]
    chosen.extend(row for row in rows[:2] if row['name'] != best['name'])
    for row in chosen:
        print(f"Test {row['name']}: MSE={mse(y, model_predict(row['model'], X)):.6f}")


def limits_demo():
    X = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    y = np.array([0., 1., 1., 0.])
    tree = SmallTree('binary', 2, 1).fit(X, y)
    print('Pure XOR gains:', [row['gain'] for row in split_candidates(X, y, 'binary')])
    print('Strict-positive-gain greedy tree stops at root:', tree.root.feature is None)
    print('Leaf probabilities:', tree.predict(X), '; this is an algorithmic limitation.')
    # A deliberately specified two-level rule can represent XOR.
    manual = (X[:, 0] > .5) != (X[:, 1] > .5)
    print('Manual two-level rule labels:', manual.astype(int))


def main():
    demos = {'classification': classification_demo, 'regression': regression_demo,
             'boosting': boosting_demo, 'comparison': comparison_demo, 'limits': limits_demo}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=[*demos, 'all'], default='all')
    args = parser.parse_args()
    np.set_printoptions(precision=6, suppress=True)
    for name, demo in demos.items():
        if args.mode in (name, 'all'):
            print(f'\n--- {name} ---'); demo()


if __name__ == '__main__':
    main()
