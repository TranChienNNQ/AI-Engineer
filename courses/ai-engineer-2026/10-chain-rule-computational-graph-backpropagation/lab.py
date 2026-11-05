"""Lesson 10: manual reverse-mode derivatives and end-to-end MLP training.

NumPy float64, one hidden layer, one binary logit, ReLU or tanh.
All parameter gradients are computed explicitly; no autograd engine is used.
XOR is a four-row training truth table, not a held-out evaluation.
"""

import argparse
import math

import numpy as np

KEYS = ('W1', 'b1', 'W2', 'b2')


def finite(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError('Use finite numeric values.')
    return values


def validate_params(params):
    if set(params) != set(KEYS):
        raise ValueError('Parameters must have exactly W1,b1,W2,b2.')
    p = {key: finite(params[key]) for key in KEYS}
    if p['W1'].ndim != 2 or not all(p['W1'].shape):
        raise ValueError('Use nonempty W1=(d,h).')
    d, h = p['W1'].shape
    if p['b1'].shape != (h,) or p['W2'].shape != (h, 1) or p['b2'].shape != (1,):
        raise ValueError('Use b1=(h,), W2=(h,1), b2=(1,).')
    return p


def clone(params):
    return {key: value.copy() for key, value in validate_params(params).items()}


def sigmoid(z):
    z = finite(z)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1/(1+np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z/(1+exp_z)
    return result


def forward(X, params, activation='relu'):
    p, X = validate_params(params), finite(X)
    if X.ndim != 2 or X.shape[1] != p['W1'].shape[0]:
        raise ValueError('Use X=(n,d) with the fitted feature order.')
    if activation not in ('relu', 'tanh'):
        raise ValueError('Use relu or tanh for the hidden activation.')
    Z1 = X@p['W1']+p['b1']
    H = np.maximum(0, Z1) if activation == 'relu' else np.tanh(Z1)
    Z2 = H@p['W2']+p['b2']
    return Z2, {'X': X, 'Z1': Z1, 'H': H, 'Z2': Z2}


def objective_value(X, y, params, activation='relu', l2=0., reduction='mean'):
    """Forward plus loss only, with no gradient calculation or parameter updates."""
    p, y = validate_params(params), finite(y)
    logits, cache = forward(X, p, activation)
    if y.ndim != 1 or not y.size or logits.shape != (len(y), 1) or not np.all((y == 0) | (y == 1)):
        raise ValueError('Use matching nonempty binary y=(n,) and logits=(n,1).')
    if not math.isfinite(l2) or l2 < 0 or reduction not in ('mean', 'sum'):
        raise ValueError('Use finite nonnegative L2 and mean or sum reduction.')
    target = y[:, None]
    losses = np.logaddexp(0, np.where(target == 1, -logits, logits))
    loss = float(losses.mean() if reduction == 'mean' else losses.sum())
    penalty = .5*l2*sum(float(np.sum(p[key]**2)) for key in ('W1', 'W2'))
    cache['target'] = target
    return loss, loss+penalty, cache


def loss_and_gradients(X, y, params, activation='relu', l2=0., reduction='mean'):
    """Return data loss, objective, parameter gradients and intermediate trace.

    J = reduced BCE + l2/2 * (sum(W1**2)+sum(W2**2)); biases unpenalized.
    The batch factor is applied once, at dZ2. ReLU'(0)=0 is a convention.
    No parameters are modified by this routine.
    """
    p = validate_params(params)
    loss, objective, cache = objective_value(X, y, p, activation, l2, reduction)
    logits, target = cache['Z2'], cache['target']
    dZ2 = (sigmoid(logits)-target)/(len(target) if reduction == 'mean' else 1)
    grads = {'W2': cache['H'].T@dZ2+l2*p['W2'], 'b2': dZ2.sum(axis=0)}
    dH = dZ2@p['W2'].T  # Use the same W2 as the forward pass, before any update.
    slope = (cache['Z1'] > 0).astype(float) if activation == 'relu' else 1-cache['H']**2
    dZ1 = dH*slope
    grads['W1'] = cache['X'].T@dZ1+l2*p['W1']
    grads['b1'] = dZ1.sum(axis=0)
    cache.update(dZ2=dZ2, dH=dH, dZ1=dZ1, dX=dZ1@p['W1'].T)
    return loss, objective, {key: grads[key] for key in KEYS}, cache


def sgd_step(params, grads, lr=.1):
    """Create all new parameters from one snapshot of parameters and gradients."""
    p = validate_params(params)
    if set(grads) != set(KEYS) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Use all four gradients and finite positive learning rate.')
    result = {}
    for key in KEYS:
        gradient = finite(grads[key])
        if gradient.shape != p[key].shape:
            raise ValueError(f'Gradient shape mismatch for {key}.')
        result[key] = p[key]-lr*gradient
    return result


def hand_example():
    # Exactly the two-example ReLU network from lesson 9.
    X, y = np.array([[1., 2.], [-1., 1.]]), np.array([1., 0.])
    p = {'W1': np.array([[1., -1.], [.5, 1.]]), 'b1': np.array([-1., 0.]),
         'W2': np.array([[2.], [-1.]]), 'b2': np.array([.5])}
    return X, y, p


def scalar_example(x=2., w=.5, b=1.):
    u = w*x; a = u+b; v = a*a; r = 3*a; loss = v+r
    da_from_square, da_from_linear = 2*a, 3.
    da = da_from_square+da_from_linear
    return {'u': u, 'a': a, 'v': v, 'r': r, 'L': loss,
            'da_square': da_from_square, 'da_linear': da_from_linear,
            'da': da, 'dw': da*x, 'db': da, 'dx': da*w}


def gradient_check(X, y, params, activation='relu', l2=.01, epsilon=1e-6):
    """Centered finite differences; report potential ReLU mask crossings.

    Cost: 2*P objective evaluations plus the analytic backward pass.
    At a ReLU kink, disagreement is not by itself proof of a backward bug.
    """
    p = clone(params)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Use a finite positive epsilon.')
    _, _, analytic, base = loss_and_gradients(X, y, p, activation, l2)
    rows = []
    for key in KEYS:
        numeric = np.zeros_like(p[key]); crossed = False
        for index in np.ndindex(p[key].shape):
            plus, minus = clone(p), clone(p)
            plus[key][index] += epsilon; minus[key][index] -= epsilon
            _, jp, cp = objective_value(X, y, plus, activation, l2)
            _, jm, cm = objective_value(X, y, minus, activation, l2)
            numeric[index] = (jp-jm)/(2*epsilon)
            if activation == 'relu':
                base_mask = base['Z1'] > 0
                crossed |= bool(np.any((cp['Z1'] > 0) != base_mask) or np.any((cm['Z1'] > 0) != base_mask))
        delta = numeric-analytic[key]
        denominator = max(1e-12, float(np.linalg.norm(numeric)+np.linalg.norm(analytic[key])))
        rows.append({'name': key, 'numeric': numeric, 'analytic': analytic[key],
                     'max_abs': float(np.max(np.abs(delta))),
                     'relative': float(np.linalg.norm(delta)/denominator),
                     'crossed_relu_kink': crossed})
    return rows


def init_params(d=2, hidden=4, seed=2026):
    if not isinstance(d, int) or not isinstance(hidden, int) or min(d, hidden) < 1:
        raise ValueError('Use positive integer layer widths.')
    rng = np.random.default_rng(seed)
    # Explicit demonstration scale, not a claim of optimal initialization.
    return {'W1': rng.normal(0, 1/math.sqrt(d), (d, hidden)), 'b1': np.zeros(hidden),
            'W2': rng.normal(0, 1/math.sqrt(hidden), (hidden, 1)), 'b2': np.zeros(1)}


def xor_data():
    return np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]), np.array([0., 1., 1., 0.])


def train_xor(steps=5000, lr=1., l2=.001, seed=2026, activation='tanh'):
    if not isinstance(steps, int) or steps < 0:
        raise ValueError('Use nonnegative integer steps.')
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('Use a finite positive learning rate.')
    X, y = xor_data(); params = init_params(seed=seed); initial = clone(params)
    history = []
    for step in range(steps+1):
        loss, objective, grads, cache = loss_and_gradients(X, y, params, activation, l2)
        accuracy = float(np.mean((cache['Z2'][:, 0] >= 0) == y))
        history.append((step, loss, objective, accuracy))
        if step < steps:
            params = sgd_step(params, grads, lr)
    return {'params': params, 'initial': initial, 'activation': activation, 'history': np.array(history),
            'l2': l2, 'lr': lr, 'seed': seed}


def scalar_demo():
    print('Shared a feeds a*a and 3*a. Reverse contributions must be added.')
    for key, value in scalar_example().items(): print(key, value)


def batch_demo():
    X, y, p = hand_example()
    loss, obj, grads, cache = loss_and_gradients(X, y, p)
    print('Forward BCE:', loss, '; objective:', obj)
    for key in ('Z1', 'H', 'Z2', 'dZ2', 'dH', 'dZ1', 'dX'): print(key, cache[key])
    for key in KEYS: print('d'+key, grads[key])


def step_demo():
    X, y, p = hand_example()
    old, _, grads, _ = loss_and_gradients(X, y, p)
    updated = sgd_step(p, grads, lr=.1)
    new, _, _, cache = loss_and_gradients(X, y, updated)
    print('One simultaneous step, lr=0.1, L2=0: old BCE=', old, '; new BCE=', new)
    for key in KEYS: print(key, updated[key])
    print('New logits:', cache['Z2'][:, 0])


def check_demo():
    X, y, p = hand_example()
    for activation in ('relu', 'tanh'):
        print('Checking all parameters with hidden', activation, 'and L2=0.01')
        for row in gradient_check(X, y, p, activation, .01):
            print(f"{row['name']}: max_abs={row['max_abs']:.3e}; relative={row['relative']:.3e}; kink={row['crossed_relu_kink']}")


def reduction_demo():
    X, y, p = hand_example()
    for reduction in ('mean', 'sum'):
        loss, _, grads, _ = loss_and_gradients(X, y, p, reduction=reduction)
        print(reduction, 'BCE:', loss, '; W1 gradient:', grads['W1'], '; b1 gradient:', grads['b1'])
    print('At L2=0, sum loss/gradients equal n times mean. With L2 fixed, penalty gradients do not scale.')


def train_demo():
    X, y = xor_data(); result = train_xor()
    p = sigmoid(forward(X, result['params'], result['activation'])[0][:, 0])
    print('End-to-end 2->4->1 tanh; seed=2026, steps=5000, lr=1, L2=0.001')
    print('First row step/data loss/objective/accuracy:', result['history'][0])
    print('Last row step/data loss/objective/accuracy:', result['history'][-1])
    print('Probabilities:', p, '; labels:', (p >= .5).astype(int))
    for key in KEYS:
        print('Parameter change norm', key, np.linalg.norm(result['params'][key]-result['initial'][key]))
    print('All scores are TRAIN on four constructed rows; no validation/test or generalization claim.')


def main():
    demos = {'scalar': scalar_demo, 'batch': batch_demo, 'step': step_demo,
             'check': check_demo, 'reduction': reduction_demo, 'train': train_demo}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=[*demos, 'all'], default='all')
    args = parser.parse_args(); np.set_printoptions(precision=9, suppress=True)
    for name, demo in demos.items():
        if args.mode in (name, 'all'):
            print(f'\n--- {name} ---'); demo()


if __name__ == '__main__':
    main()
