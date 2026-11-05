"""Lesson 1: fit a linear model by grid search; Python standard library only.

Run: python3 lab.py --variant baseline
All datasets are synthetic teaching examples, not backend benchmarks.
"""

import argparse


def predict(x, w, b):
    """Predict processing time (ms) from batch count."""
    return w * x + b


def mse(data, w, b):
    """Return mean squared error (ms^2) for a nonempty dataset."""
    squared_errors = [
        (predict(x, w, b) - y) ** 2
        for x, y in data
    ]
    return sum(squared_errors) / len(data)


def train_grid(data, w_values, b_values):
    """Choose parameters using only the training data and candidate lists."""
    best_w, best_b = None, None
    best_loss = float("inf")
    for w in w_values:
        for b in b_values:
            loss = mse(data, w, b)
            if loss < best_loss:
                best_w, best_b, best_loss = w, b, loss
    return best_w, best_b, best_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["baseline", "no-bias", "noisy"],
        default="baseline",
    )
    args = parser.parse_args()

    train = [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)]
    held_out = [(4.0, 9.0)]
    w_values = [i / 10 for i in range(41)]
    b_values = [i / 10 for i in range(-20, 21)]

    if args.variant == "no-bias":
        b_values = [0.0]
    elif args.variant == "noisy":
        train = [(1.0, 3.2), (2.0, 4.7), (3.0, 7.1)]

    print(f"Variant: {args.variant}")
    print(f"Initial train MSE: {mse(train, 1.0, 0.0):.4f}")
    w, b, loss = train_grid(train, w_values, b_values)
    print(f"Best w: {w:.2f}; best b: {b:.2f}")
    print(f"Train MSE: {loss:.4f}")

    # The held-out target is not passed to the fitting function.
    print(f"Held-out prediction at x=4: {predict(4.0, w, b):.2f} ms")
    print(f"Held-out MSE: {mse(held_out, w, b):.4f}")


if __name__ == "__main__":
    main()
