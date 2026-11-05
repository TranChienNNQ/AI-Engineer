"""Lesson 2: derivatives and full-batch gradient descent, standard library only.

Examples:
  python3 lab.py --mode toy --lr 0.1 --steps 5
  python3 lab.py --mode linear --lr 0.1 --steps 1000
  python3 lab.py --mode check
  python3 lab.py --mode compare
"""

import argparse
import math


TRAIN = [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)]
HELD_OUT = [(4.0, 9.0)]


def toy_loss(w):
    return (w - 2.0) ** 2


def toy_derivative(w):
    return 2.0 * (w - 2.0)


def train_toy(lr, steps, start=5.0):
    """Return (step, w, loss) before training and after every update."""
    w = start
    history = [(0, w, toy_loss(w))]
    for step in range(1, steps + 1):
        w = w - lr * toy_derivative(w)
        loss = toy_loss(w)
        if not math.isfinite(loss):
            raise ValueError("Non-finite loss; reduce learning rate or steps.")
        history.append((step, w, loss))
    return history


def predict(x, w, b):
    return w * x + b


def mse(data, w, b):
    return sum((predict(x, w, b) - y) ** 2 for x, y in data) / len(data)


def gradients(data, w, b):
    """Partial derivatives of MSE, evaluated at the SAME (w, b)."""
    n = len(data)
    dw = 2.0 / n * sum((predict(x, w, b) - y) * x for x, y in data)
    db = 2.0 / n * sum(predict(x, w, b) - y for x, y in data)
    return dw, db


def train_linear(data, lr, steps, start_w=1.0, start_b=0.0):
    """Full-batch GD; only the supplied training data influences updates."""
    w, b = start_w, start_b
    history = [(0, w, b, mse(data, w, b))]
    for step in range(1, steps + 1):
        dw, db = gradients(data, w, b)
        # Both gradients were computed before changing either parameter.
        w, b = w - lr * dw, b - lr * db
        loss = mse(data, w, b)
        if not math.isfinite(loss):
            raise ValueError("Non-finite loss; reduce learning rate or steps.")
        history.append((step, w, b, loss))
    return history


def numerical_gradients(data, w, b, h=1e-5):
    """Central differences: keep the other parameter fixed in each check."""
    dw = (mse(data, w + h, b) - mse(data, w - h, b)) / (2 * h)
    db = (mse(data, w, b + h) - mse(data, w, b - h)) / (2 * h)
    return dw, db


def sampled_rows(history):
    for row in history:
        if row[0] <= 5 or row[0] == history[-1][0]:
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["toy", "linear", "check", "compare"], default="linear")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")

    if args.mode == "check":
        print("point (w=1, b=0)")
        analytic = gradients(TRAIN, 1.0, 0.0)
        numeric = numerical_gradients(TRAIN, 1.0, 0.0)
        for name, a, n in zip(("dw", "db"), analytic, numeric):
            print(f"{name}: analytic={a:.8f}; numerical={n:.8f}; abs_diff={abs(a-n):.2e}")
        return

    if args.mode == "compare":
        print("Toy loss; start w=5; 12 updates")
        print("lr       final_w       final_loss")
        for lr in (0.01, 0.25, 1.0, 1.1):
            _, w, loss = train_toy(lr, 12)[-1]
            print(f"{lr:<8.2f} {w:>12.6f} {loss:>16.6e}")
        return

    try:
        if args.mode == "toy":
            history = train_toy(args.lr, args.steps)
            print("step          w           loss")
            for step, w, loss in sampled_rows(history):
                print(f"{step:>4} {w:>12.6f} {loss:>14.6e}")
        else:
            history = train_linear(TRAIN, args.lr, args.steps)
            print("step          w            b           train_mse")
            for step, w, b, loss in sampled_rows(history):
                print(f"{step:>4} {w:>12.6f} {b:>12.6f} {loss:>16.6e}")
            _, w, b, _ = history[-1]
            print(f"Held-out prediction at x=4: {predict(4.0, w, b):.6f} ms")
            print(f"Held-out MSE: {mse(HELD_OUT, w, b):.6e}")
    except (ValueError, OverflowError) as error:
        parser.exit(1, f"Training stopped: {error}\n")


if __name__ == "__main__":
    main()
