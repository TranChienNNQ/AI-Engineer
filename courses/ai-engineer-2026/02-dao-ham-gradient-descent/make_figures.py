"""Regenerate the lesson's mathematical plots as PNG and SVG.

Install requirements-figures.txt in a virtual environment, then run this file.
Figures are written beside this script, regardless of working directory.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab import TRAIN, train_linear, train_toy


OUT = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, GREEN, RED = "#2563eb", "#d97706", "#15803d", "#dc2626"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.17,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def derivative_figure():
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    w = np.linspace(-0.5, 6, 500)
    ax.plot(w, (w - 2) ** 2, color=BLUE, linewidth=2.5, label=r"$f(w)=(w-2)^2$")
    tangent_x = np.linspace(3.6, 6, 150)
    ax.plot(tangent_x, 9 + 6 * (tangent_x - 5), color=ORANGE, linewidth=2,
            label=r"Tiếp tuyến tại $w=5$: độ dốc $6$")
    secant_x = np.linspace(4.7, 6, 100)
    ax.plot(secant_x, 9 + 6.8 * (secant_x - 5), "--", color=GREEN,
            label=r"Cát tuyến từ $5$ đến $5.8$: độ dốc $6.8$")
    ax.scatter([2, 5, 5.8], [0, 9, 14.44], color=[BLUE, ORANGE, GREEN], zorder=5)
    ax.annotate("Cực tiểu: (2, 0)", (2, 0), (0.15, 3.2),
                arrowprops={"arrowstyle": "->", "color": BLUE}, color=BLUE)
    ax.annotate("(5, 9)", (5, 9), (3.5, 10),
                arrowprops={"arrowstyle": "->", "color": ORANGE}, color=ORANGE)
    ax.set(title="Đạo hàm mô tả độ dốc tại một điểm",
           xlabel="Tham số w (không đơn vị)", ylabel="Giá trị f(w) (không đơn vị)",
           xlim=(-0.5, 6), ylim=(-1, 18))
    ax.legend(loc="upper left", fontsize=10)
    save(fig, "01-dao-ham")


def rates_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), layout="constrained")
    colors = [BLUE, GREEN, ORANGE, RED]
    for lr, color in zip((0.01, 0.25, 1.0, 1.1), colors):
        h = np.array(train_toy(lr, 12))
        label = f"α = {lr:g}"
        axes[0].plot(h[:, 0], h[:, 1], "o-", ms=3.5, color=color, label=label)
        axes[1].semilogy(h[:, 0], h[:, 2], "o-", ms=3.5, color=color, label=label)
    axes[0].axhline(2, color="#475569", linestyle="--", linewidth=1, label="Nghiệm w = 2")
    axes[0].set(title="Tham số sau mỗi cập nhật", xlabel="Số lần cập nhật", ylabel="w")
    axes[1].set(title="Loss: hội tụ, đứng yên hoặc tăng", xlabel="Số lần cập nhật",
                ylabel="f(w) — trục logarit")
    for ax in axes:
        ax.legend(fontsize=9)
    fig.suptitle(r"Cùng $f(w)=(w-2)^2$, cùng $w_0=5$, khác learning rate", fontsize=15)
    save(fig, "02-learning-rate")


def linear_figure():
    history = np.array(train_linear(TRAIN, 0.1, 1000))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), layout="constrained")
    ws = np.linspace(0.9, 2.48, 250)
    bs = np.linspace(-0.12, 1.15, 250)
    W, B = np.meshgrid(ws, bs)
    Z = sum((W * x + B - y) ** 2 for x, y in TRAIN) / len(TRAIN)
    cs = axes[0].contour(W, B, Z, levels=[0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                         colors="#94a3b8", linewidths=0.85)
    axes[0].clabel(cs, fontsize=8, fmt="%g")
    axes[0].plot(history[:, 1], history[:, 2], color=BLUE, linewidth=1.8)
    for step, offset in [(0, (12, 8)), (1, (-85, 28)), (2, (-100, -25)), (20, (-90, -10))]:
        _, w, b, _ = history[step]
        axes[0].scatter(w, b, color=BLUE, s=22, zorder=5)
        axes[0].annotate(f"Bước {step}", (w, b), xytext=offset, textcoords="offset points",
                         fontsize=9, arrowprops={"arrowstyle": "->", "color": BLUE})
    axes[0].scatter(2, 1, marker="*", s=170, color=ORANGE, zorder=6)
    axes[0].annotate("Nghiệm (2, 1)", (2, 1), xytext=(-105, 15), textcoords="offset points",
                     color=ORANGE, fontsize=10)
    axes[0].set(title="Đường đi trên mặt phẳng tham số",
                xlabel="w (ms/lô)", ylabel="b (ms)", xlim=(0.9, 2.48), ylim=(-0.12, 1.15))
    axes[1].semilogy(history[:, 0], history[:, 3], color=BLUE, linewidth=2)
    axes[1].set(title="MSE trên tập huấn luyện", xlabel="Số lần cập nhật",
                ylabel="MSE (ms²) — trục logarit")
    fig.suptitle("Hồi quy bài 1: gradient descent với α = 0.1", fontsize=15)
    save(fig, "03-hoi-quy")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    derivative_figure()
    rates_figure()
    linear_figure()
    print(f"Generated 3 PNG and 3 SVG figures in {OUT}")


if __name__ == "__main__":
    main()
