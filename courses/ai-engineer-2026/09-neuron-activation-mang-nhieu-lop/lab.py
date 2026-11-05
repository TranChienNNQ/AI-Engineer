"""Lesson 9: neurons, activations and forward computation in small MLPs.

NumPy only. Hand-specified hidden weights demonstrate representation.
The only fitting routine trains a binary output layer on FIXED features,
using the logistic gradient from lesson 6. It does not backpropagate through
hidden layers. XOR is a four-row teaching truth table, not held-out evidence.
"""

import argparse
import math

import numpy as np


def finite_array(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('Use finite numeric inputs.')
    return values


def sigmoid(z):
    z = finite_array(z)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1/(1+np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z/(1+exp_z)
    return result


def activate(z, name='relu'):
    z = finite_array(z)
    if name == 'identity':
        return z.copy()
    if name == 'relu':
        return np.maximum(0, z)
    if name == 'sigmoid':
        return sigmoid(z)
    if name == 'tanh':
        return np.tanh(z)
    if name == 'leaky_relu':
        return np.where(z >= 0, z, .1*z)
    if name == 'gelu':
        # Exact-form GELU z*Phi(z), evaluated using the scalar standard-library erf.
        # This simple implementation is for illustration, not an optimized kernel.
        erf = np.array([math.erf(float(v)/math.sqrt(2)) for v in z.ravel()]).reshape(z.shape)
        return .5*z*(1+erf)
    raise ValueError('Use identity, relu, sigmoid, tanh, leaky_relu or gelu.')


def dense(X, W, b):
    """Row-batch convention: X=(n,d), W=(d,h), b=(h,)."""
    X, W, b = map(finite_array, (X, W, b))
    if (X.ndim != 2 or W.ndim != 2 or b.ndim != 1 or X.shape[1] == 0
            or X.shape[1] != W.shape[0] or W.shape[1] == 0 or b.shape != (W.shape[1],)):
        raise ValueError('Expected X=(n,d), W=(d,h), b=(h,), with nonempty feature axes.')
    return X@W+b


def forward(X, layers):
    """Each layer is (W,b,activation); output is raw unless configured otherwise.

    Returns a trace with each Z (pre-activation) and A (post-activation).
    The trace is for inspection here, not an automatic differentiation graph.
    """
    X = finite_array(X)
    if X.ndim != 2 or not layers:
        raise ValueError('Use X=(n,d) and at least one layer.')
    A, trace = X, []
    for W, b, activation in layers:
        Z = dense(A, W, b)
        A = activate(Z, activation)
        trace.append({'Z': Z, 'A': A, 'activation': activation})
    return A, trace


def parameter_count(layers):
    total, previous = 0, None
    if not layers:
        raise ValueError('At least one layer is required.')
    for W, b, activation in layers:
        W, b = finite_array(W), finite_array(b)
        if W.ndim != 2 or not all(W.shape) or b.shape != (W.shape[1],):
            raise ValueError('Use W=(input,output) and b=(output,) per layer.')
        if previous is not None and previous != W.shape[0]:
            raise ValueError('Neighboring layer widths must match.')
        activate(np.zeros(1), activation)  # Validate the activation name.
        total += W.size+b.size
        previous = W.shape[1]
    return total


def collapse_affine(layers):
    """Combine identity-only dense layers into one W,b, preserving row convention."""
    parameter_count(layers)
    if any(name != 'identity' for _, _, name in layers):
        raise ValueError('Only identity activations may be collapsed by this function.')
    W, b, _ = layers[0]
    W, b = finite_array(W).copy(), finite_array(b).copy()
    for next_W, next_b, _ in layers[1:]:
        b = b@next_W+next_b
        W = W@next_W
    return W, b


def bce_logits(y, z):
    y, z = finite_array(y), finite_array(z)
    if y.ndim != 1 or not y.size or y.shape != z.shape or not np.all((y == 0) | (y == 1)):
        raise ValueError('Use matching nonempty 1-D binary targets and logits.')
    return float(np.mean(np.logaddexp(0, np.where(y == 1, -z, z))))


def softmax(z):
    z = finite_array(z)
    if z.ndim < 1 or z.shape[-1] == 0:
        raise ValueError('Use a nonempty final class axis.')
    exp_z = np.exp(z-z.max(axis=-1, keepdims=True))
    return exp_z/exp_z.sum(axis=-1, keepdims=True)


def hand_example():
    X = np.array([[1., 2.], [-1., 1.]])
    W1, b1 = np.array([[1., -1.], [.5, 1.]]), np.array([-1., 0.])
    W2, b2 = np.array([[2.], [-1.]]), np.array([.5])
    return X, [(W1, b1, 'relu'), (W2, b2, 'identity')]


def xor_data():
    return np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]), np.array([0., 1., 1., 0.])


def xor_layers():
    W1, b1 = np.array([[1., -1.], [-1., 1.]]), np.zeros(2)
    W2, b2 = np.array([[4.], [4.]]), np.array([-2.])
    return [(W1, b1, 'relu'), (W2, b2, 'identity')]


def xor_features(X):
    W, b, name = xor_layers()[0]
    return activate(dense(X, W, b), name)


def readout_objective(H, y, w, b, l2=.01):
    """Fixed-feature binary head: mean BCE + l2/2 * ||w||^2, no bias penalty."""
    H, y, w = map(finite_array, (H, y, w))
    if H.ndim != 2 or y.ndim != 1 or w.shape != (H.shape[1],) or len(y) != len(H):
        raise ValueError('Use H=(n,h), y=(n,), w=(h,).')
    if not math.isfinite(b) or not math.isfinite(l2) or l2 < 0:
        raise ValueError('Use finite bias and finite nonnegative L2.')
    z = H@w+b
    loss = bce_logits(y, z)
    error = sigmoid(z)-y
    return (loss, loss+.5*l2*float(w@w), H.T@error/len(y)+l2*w, float(error.mean()))


def fit_readout(H, y, steps=3000, lr=.2, l2=.01):
    """Fit only w,b of a logistic head. H is fixed and is never updated."""
    H, y = finite_array(H), finite_array(y)
    if H.ndim != 2 or H.shape[1] == 0:
        raise ValueError('Use nonempty features H=(n,h).')
    if not isinstance(steps, (int, np.integer)) or steps < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Use nonnegative integer steps and finite positive learning rate.')
    w, b, history = np.zeros(H.shape[1]), 0., []
    for step in range(steps+1):
        loss, objective, dw, db = readout_objective(H, y, w, b, l2)
        history.append((step, loss, objective))
        if step < steps:
            w, b = w-lr*dw, b-lr*db
    return {'w': w, 'b': b, 'l2': l2, 'history': np.array(history)}


def readout_experiment():
    X, y = xor_data()
    return {'raw': fit_readout(X, y), 'fixed_relu': fit_readout(xor_features(X), y)}


def neuron_demo():
    X = np.array([[2., -1.]])
    W, b = np.array([[.5], [-2.]]), np.array([-.5])
    z = dense(X, W, b)
    print('z = 2*0.5 + (-1)*(-2) - 0.5 =', z.item())
    for name in ['identity', 'relu', 'sigmoid', 'tanh']:
        print(name, activate(z, name).item())


def activations_demo():
    z = np.array([-2., -1., 0., 1., 2.])
    print('z:', z)
    for name in ['identity', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'gelu']:
        print(name, activate(z, name))


def forward_demo():
    X, layers = hand_example()
    logits, trace = forward(X, layers)
    print('X:', X)
    for i, item in enumerate(trace, 1):
        print(f'layer {i}: Z shape={item["Z"].shape}, A shape={item["A"].shape}')
        print('Z:', item['Z']); print('A:', item['A'])
    print('Total parameters:', parameter_count(layers))
    print('Probabilities:', sigmoid(logits[:, 0]))
    print('BCE for y=[1,0]:', bce_logits(np.array([1., 0.]), logits[:, 0]))


def collapse_demo():
    X, layers = hand_example()
    affine = [(W, b, 'identity') for W, b, _ in layers]
    W, b = collapse_affine(affine)
    print('Collapsed W:', W, '; b:', b)
    print('Identity-only stack:', forward(X, affine)[0].ravel())
    print('Collapsed model:', dense(X, W, b).ravel())
    print('With hidden ReLU:', forward(X, layers)[0].ravel())


def xor_demo():
    X, y = xor_data()
    logits, trace = forward(X, xor_layers())
    p = sigmoid(logits[:, 0])
    print('Hidden features:', trace[0]['A'])
    print('Hand-set logits:', logits.ravel(), '; p:', p)
    print('Labels:', (p >= .5).astype(int), '; truth-table accuracy:', np.mean((p >= .5) == y))
    print('BCE:', bce_logits(y, logits[:, 0]))
    print('All weights above are specified, not learned. No held-out score is reported.')


def readout_demo():
    X, y = xor_data()
    for name, model in readout_experiment().items():
        H = X if name == 'raw' else xor_features(X)
        p = sigmoid(H@model['w']+model['b'])
        print(name, 'w:', model['w'], '; b:', model['b'])
        print('p:', p, '; training accuracy:', np.mean((p >= .5) == y))
        print('BCE and penalized objective:', model['history'][-1, 1:])
    print('Hidden W1,b1 stay fixed; this is not end-to-end MLP training.')


def heads_demo():
    z = np.log(np.array([[1., 2., 3.], [3., 2., 1.]]))
    print('Example logits:', z)
    print('Multiclass softmax, one class per row:', softmax(z))
    print('Multilabel independent sigmoids:', sigmoid(z))
    print('Regression identity output can be negative:', activate(np.array([-2., 3.]), 'identity'))


def main():
    demos = {'neuron': neuron_demo, 'activations': activations_demo, 'forward': forward_demo,
             'collapse': collapse_demo, 'xor': xor_demo, 'readout': readout_demo, 'heads': heads_demo}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=[*demos, 'all'], default='all')
    args = parser.parse_args()
    np.set_printoptions(precision=6, suppress=True)
    for name, demo in demos.items():
        if args.mode in (name, 'all'):
            print(f'\n--- {name} ---'); demo()


if __name__ == '__main__':
    main()
