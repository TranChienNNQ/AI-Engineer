"""Recreate the lesson's diagrams from the same arrays as lab.py."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from lab import example_data


OUT = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, GREEN = "#2563eb", "#d97706", "#15803d"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def vector_figure():
    fig, ax = plt.subplots(figsize=(8.3, 6.4), layout="constrained")
    a, b = np.array([2, 1]), np.array([1, 2])
    for origin, delta, color, style in [([0, 0], a, BLUE, "-"),
                                        (a, b, ORANGE, "-"),
                                        ([0, 0], a+b, GREEN, "-")]:
        ax.annotate("", xy=np.array(origin)+delta, xytext=origin,
                    arrowprops={"arrowstyle": "->", "lw": 2.7, "color": color, "linestyle": style})
    ax.plot([0, b[0]], [0, b[1]], "--", color=ORANGE, alpha=.65)
    ax.scatter([0, 2, 3], [0, 1, 3], s=25, color="#334155")
    ax.text(1.02, .32, "a = (2, 1)", color=BLUE)
    ax.text(2.58, 1.75, "b = (1, 2)", color=ORANGE)
    ax.text(.15, 1.75, "b tại gốc", color=ORANGE)
    ax.text(1.05, 2.05, "a + b = (3, 3)", color=GREEN)
    ax.set(xlim=(-.2, 3.8), ylim=(-.2, 3.6), aspect="equal",
           xlabel="Tọa độ thứ nhất", ylabel="Tọa độ thứ hai",
           title="Cộng vector: cộng từng thành phần, ghép đầu–đuôi")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.grid(alpha=.2)
    save(fig, "01-vector")


def matrix_table(ax, data, title, highlighted=None, color="#dbeafe"):
    ax.axis("off")
    table = ax.table(cellText=[[f"{v:g}" for v in row] for row in data],
                     cellLoc="center", loc="center", bbox=[.08, .2, .84, .59])
    table.auto_set_font_size(False); table.set_fontsize(19)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#94a3b8")
        cell.set_facecolor(color if highlighted and key in highlighted else "#f8fafc")
    ax.set_title(title, fontsize=13, pad=12)


def matmul_figure():
    X, _ = example_data()
    W = np.array([[2., 1.], [1., -1.]])
    fig = plt.figure(figsize=(12.2, 4.6), layout="constrained")
    grid = fig.add_gridspec(1, 5, width_ratios=[3, .35, 2, .35, 3])
    matrix_table(fig.add_subplot(grid[0, 0]), X, "X: 3 hàng × 2 cột", {(0, 0), (0, 1)})
    matrix_table(fig.add_subplot(grid[0, 2]), W, "W: 2 hàng × 2 cột", {(0, 0), (1, 0)}, "#ffedd5")
    matrix_table(fig.add_subplot(grid[0, 4]), X@W, "C = XW: 3 hàng × 2 cột", {(0, 0)}, "#dcfce7")
    for col, sign in [(1, "@"), (3, "=")]:
        ax=fig.add_subplot(grid[0, col]); ax.axis("off")
        ax.text(.5, .5, sign, ha="center", va="center", fontsize=28)
    fig.suptitle("Nhân ma trận: một hàng nhân một cột rồi cộng", fontsize=17, fontweight="bold")
    fig.supxlabel("Ô tô màu ở kết quả: C[0,0] = 1 × 2 + 2 × 1 = 4", fontsize=13)
    save(fig, "02-nhan-ma-tran")


def broadcasting_figure():
    _, y = example_data()
    correct = y - y
    wrong = y[:, None] - y
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.5), layout="constrained")
    for ax, data, title in [(axes[0], correct[:, None], "Đúng: (3,) − (3,) → (3,)"),
                            (axes[1], wrong, "Sai ý định: (3,1) − (3,) → (3,3)")]:
        ax.imshow(data, cmap="RdBu", vmin=-4, vmax=4, aspect="equal")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v=data[i,j]
                ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=20,
                        color="white" if abs(v)>=3 else "#0f172a")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_yticks(range(3), ["Mẫu 0", "Mẫu 1", "Mẫu 2"])
        if data.shape[1]==1:
            ax.set_xticks([0], ["Sai số từng mẫu"])
        else:
            ax.set_xticks(range(3), ["Đích 0", "Đích 1", "Đích 2"])
        ax.set_xlabel(f"MSE = {np.mean(data**2):.4f} ms²")
    fig.suptitle("Cùng dự đoán hoàn hảo [5, 8, 9], khác shape → khác phép so sánh", fontsize=15)
    save(fig, "03-broadcasting")


def tensor_figure():
    images = np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.7), layout="constrained")
    cmaps=["Reds", "Greens", "Blues"]
    for batch in range(2):
        for channel in range(3):
            ax=axes[batch, channel]
            data=images[batch,:,:,channel]
            ax.imshow(data, cmap=cmaps[channel], vmin=0, vmax=45)
            for h in range(2):
                for w in range(3):
                    ax.text(w,h,str(data[h,w]),ha="center",va="center",fontsize=16,
                            color="white" if data[h,w]>24 else "#0f172a")
            ax.set_xticks(range(3)); ax.set_yticks(range(2))
            ax.set_xlabel("Cột pixel W"); ax.set_ylabel("Hàng pixel H")
            ax.set_title(f"Ảnh {batch}, kênh {channel}", fontsize=12)
            if batch==1 and channel==1:
                ax.add_patch(Rectangle((1.5,-.5),1,1,fill=False,edgecolor=ORANGE,linewidth=3))
    fig.suptitle("Tensor (N,H,W,C) = (2,2,3,3): tách theo ảnh và kênh để quan sát", fontsize=15, fontweight="bold")
    fig.supxlabel("Ô 25: NHWC[1,0,2,1] = NCHW[1,1,0,2]; đổi vị trí trục, giữ ý nghĩa pixel", fontsize=12)
    save(fig, "04-tensor")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
                         "figure.facecolor":"white", "savefig.facecolor":"white",
                         "axes.spines.top":False, "axes.spines.right":False})
    vector_figure(); matmul_figure(); broadcasting_figure(); tensor_figure()
    print(f"Generated 4 PNG and 4 SVG figures in {OUT}")


if __name__ == "__main__":
    main()
