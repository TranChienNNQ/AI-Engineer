"""Lesson 3: vectors, matrices, tensor shapes and vectorized regression.

All numbers are synthetic teaching examples. NumPy is the only dependency.
Run: python3 lab.py --mode all
"""

import argparse
import math

import numpy as np


def example_data():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 2.0]], dtype=np.float64)
    y = np.array([5.0, 8.0, 9.0], dtype=np.float64)
    return X, y


def predict(X, w, b):
    """Convention: X=(n,d), w=(d,), scalar b; output=(n,)."""
    if X.ndim != 2 or w.ndim != 1 or X.shape[1] != w.shape[0]:
        raise ValueError("Expected X=(n,d) and w=(d,).")
    if np.ndim(b) != 0:
        raise ValueError("Expected scalar b.")
    return X @ w + b


def residuals(pred, y):
    """Reject silent pairwise broadcasting: one prediction per target."""
    if pred.ndim != 1 or y.ndim != 1 or pred.shape != y.shape:
        raise ValueError("Expected matching 1-D prediction and target arrays.")
    return pred - y


def mse(X, y, w, b):
    r = residuals(predict(X, w, b), y)
    return float(np.mean(r ** 2))


def gradients(X, y, w, b):
    r = residuals(predict(X, w, b), y)
    dw = (2.0 / len(y)) * (X.T @ r)
    db = float(2.0 * np.mean(r))
    return dw, db


def gradients_loop(X, y, w, b):
    """Independent explicit loops, used to compare with the vectorized formula."""
    dw = np.zeros_like(w)
    db = 0.0
    for i in range(len(y)):
        prediction = b
        for j in range(len(w)):
            prediction += X[i, j] * w[j]
        r = prediction - y[i]
        db += 2.0 * r / len(y)
        for j in range(len(w)):
            dw[j] += 2.0 * r * X[i, j] / len(y)
    return dw, db


def train(X, y, lr=0.01, steps=10000):
    """Train the lesson's two-feature example from the documented initial point."""
    w, b = np.array([1.0, 0.0]), 0.0
    for _ in range(steps):
        dw, db = gradients(X, y, w, b)
        w, b = w - lr * dw, b - lr * db
        if not np.all(np.isfinite(w)) or not math.isfinite(b):
            raise ValueError("Parameters became non-finite; reduce the learning rate.")
    return w, b


def shapes_demo():
    for name, a in [
        ("scalar", np.array(3.0)),
        ("vector", np.array([1.0, 2.0, 3.0])),
        ("row", np.array([[1.0, 2.0, 3.0]])),
        ("column", np.array([[1.0], [2.0], [3.0]])),
    ]:
        print(f"{name}: shape={a.shape}, ndim={a.ndim}, size={a.size}")
    v = np.array([1.0, 2.0, 3.0])
    print(f"v.T shape: {v.T.shape}")
    a, b = np.array([2, 1]), np.array([1, 2])
    print("a+b:", a + b, "; a*b:", a * b, "; a@b:", a @ b)
    print("norm([3,4]):", np.linalg.norm([3.0, 4.0]))


def predict_demo():
    X, y = example_data()
    w = np.array([2.0, 1.0])
    print("X shape:", X.shape, "; w shape:", w.shape)
    print("X@w:", X @ w)
    print("X@w+1:", predict(X, w, 1.0))
    print("X*w:\n", X * w)
    W = np.array([[2.0, 1.0], [1.0, -1.0]])
    print("X@W:\n", X @ W)
    print("mean(axis=0):", X.mean(axis=0))
    print("mean(axis=1):", X.mean(axis=1))
    print("MSE at w=[2,1], b=1:", mse(X, y, w, 1.0))


def gradient_demo():
    X, y = example_data()
    w, b = np.array([1.0, 0.0]), 0.0
    dw, db = gradients(X, y, w, b)
    lw, lb = gradients_loop(X, y, w, b)
    print("initial predictions:", predict(X, w, b))
    print("initial MSE:", mse(X, y, w, b))
    print("dw:", dw, "; db:", db)
    print("matches explicit loops:", np.allclose(dw, lw) and np.isclose(db, lb))
    w, b = w - 0.01 * dw, b - 0.01 * db
    print("after one update, w:", w, "; b:", b)
    print("new predictions:", predict(X, w, b))
    print("new MSE:", mse(X, y, w, b))


def broadcast_demo():
    _, y = example_data()
    pred = y.copy()  # Perfect predictions, intentionally shaped two ways below.
    correct = pred - y
    wrong = pred[:, None] - y
    print("correct residual shape:", correct.shape, "; MSE:", np.mean(correct ** 2))
    print("wrong residual shape:", wrong.shape, "; MSE:", np.mean(wrong ** 2))
    print("wrong residuals:\n", wrong)
    try:
        residuals(pred[:, None], y)
    except ValueError as error:
        print("shape guard:", error)


def tensor_demo():
    images = np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3)
    channels_first = images.transpose(0, 3, 1, 2)
    wrong = images.reshape(2, 3, 2, 3)
    print("NHWC:", images.shape, "; dtype:", images.dtype, "; nbytes:", images.nbytes)
    print("NCHW:", channels_first.shape)
    print("flatten per sample:", images.reshape(2, -1).shape)
    print("same pixel/channel:", images[1, 0, 2, 1], channels_first[1, 1, 0, 2])
    print("reshape is NOT transpose:", wrong[1, 1, 0, 2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "shapes", "predict", "gradient", "broadcast", "tensor", "train"], default="all")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    if args.steps < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Use nonnegative --steps and finite positive --lr.")
    np.set_printoptions(precision=6, suppress=True)
    if args.mode == "train":
        X, y = example_data()
        w, b = train(X, y, lr=args.lr, steps=args.steps)
        print("w:", w, "; b:", b)
        print("train MSE:", mse(X, y, w, b))
        return
    demos = {"shapes": shapes_demo, "predict": predict_demo, "gradient": gradient_demo,
             "broadcast": broadcast_demo, "tensor": tensor_demo}
    for name, demo in demos.items():
        if args.mode in ("all", name):
            print(f"\n--- {name} ---")
            demo()


if __name__ == "__main__":
    main()
