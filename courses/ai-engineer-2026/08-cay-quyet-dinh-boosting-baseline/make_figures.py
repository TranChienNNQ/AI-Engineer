"""Rebuild five lesson 8 figures from the fitted models and constructed data."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lab import (SmallTree, SquaredBoost, development_data, final_test_data,
                 hand_classification, hand_regression, model_predict, mse,
                 region_data, select_model, split_candidates)

OUT = Path(__file__).resolve().parent/'figures'
BLUE, ORANGE, GREEN, RED = '#2563eb', '#d97706', '#15803d', '#dc2626'


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=170)
    fig.savefig(OUT/f'{name}.svg')
    plt.close(fig)


def tree_diagram(ax, tree, feature_names):
    leaves = []
    def collect(node):
        if node.feature is None:
            leaves.append(node)
        else:
            collect(node.left); collect(node.right)
    collect(tree.root)
    leaf_x = {id(node): (i+.5)/len(leaves) for i, node in enumerate(leaves)}
    def layout(node, depth):
        y = .85-depth*.30
        if node.feature is None:
            x = leaf_x[id(node)]
        else:
            lx, ly = layout(node.left, depth+1)
            rx, ry = layout(node.right, depth+1)
            x = (lx+rx)/2
            for cx, cy, label in [(lx, ly, 'Có (≤)'), (rx, ry, 'Không (>)')]:
                ax.plot([x, cx], [y-.03, cy+.06], color='#64748b', lw=1.5, zorder=1)
                ax.text((x+cx)/2, (y+cy)/2, label, fontsize=9, ha='center',
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
        text = f'p={node.value:g}\nn={node.n}' if node.feature is None else (
            f'{feature_names[node.feature]} ≤ {node.threshold:g}?\nn={node.n}')
        ax.text(x, y, text, ha='center', va='center', fontsize=11, zorder=3,
                bbox={'boxstyle': 'round,pad=.5', 'facecolor': '#dcfce7' if node.feature is None else '#dbeafe',
                      'edgecolor': GREEN if node.feature is None else BLUE})
        return x, y
    layout(tree.root, 0)
    ax.set(xlim=(-.03, 1.03), ylim=(.02, 1.01))
    ax.axis('off')


def gini_figure():
    X, y = hand_classification()
    rows = split_candidates(X, y, 'binary')
    model = SmallTree('binary', 1, 1).fit(X, y)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout='constrained')
    bars = axes[0].bar([f"{r['threshold']:g}" for r in rows], [r['score'] for r in rows],
                       color=[GREEN if r['threshold'] == model.root.threshold else BLUE for r in rows])
    axes[0].bar_label(bars, labels=[f"{r['score']:.3f}" for r in rows], padding=4, fontsize=10)
    axes[0].axhline(.5, color=RED, ls='--', label='Gini gốc = 0.5')
    axes[0].set(title='Cân Gini con theo số mẫu', xlabel='Ngưỡng feature x ≤ t',
                ylabel='Gini sau chia (thấp hơn tốt hơn)', ylim=(0, .65))
    axes[0].legend(fontsize=10)
    tree_diagram(axes[1], model, ['x'])
    axes[1].set_title('Stump đã fit: chọn 2.5 khi hòa với 4.5')
    fig.suptitle('x=[1,2,3,4,5,6], y=[0,0,1,0,1,1]', fontsize=16, fontweight='bold')
    save(fig, '01-gini-stump')


def regions_figure():
    X, y = region_data(); model = SmallTree('binary', 2, 1).fit(X, y)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    tree_diagram(axes[0], model, ['Tải x₁', 'Lỗi x₂ (%)'])
    axes[0].set_title('Cây depth=2 fit từ tám điểm')
    a, b = np.meshgrid(np.linspace(.5, 4.5, 200), np.linspace(-.4, 2.4, 160))
    probabilities = model.predict(np.column_stack([a.ravel(), b.ravel()])).reshape(a.shape)
    axes[1].contourf(a, b, probabilities, levels=[-.01, .5, 1.01], colors=['#dbeafe', '#ffedd5'])
    axes[1].axvline(2.5, color='#64748b', ls='--')
    axes[1].plot([2.5, 4.5], [1, 1], color='#64748b', ls='--')
    for label, color, marker in [(0, BLUE, 'o'), (1, ORANGE, 'D')]:
        axes[1].scatter(X[y == label, 0], X[y == label, 1], color=color, marker=marker,
                        edgecolor='white', s=80, label=f'Nhãn thật {label}', zorder=3)
    axes[1].text(1.4, 1, 'p=0', ha='center', fontsize=13, color=BLUE)
    axes[1].text(3.5, 1.55, 'p=1', ha='center', fontsize=13, color=ORANGE)
    axes[1].set(title='Các vùng hằng, ranh giới song song trục', xlabel='Tải x₁ (đơn vị giả định)',
                ylabel='Tỷ lệ lỗi x₂ (%)', xlim=(.5, 4.5), ylim=(-.4, 2.4))
    axes[1].legend(loc='lower left', fontsize=9)
    fig.suptitle('Đi theo điều kiện tới lá; không phải một đường thẳng logit', fontsize=16, fontweight='bold')
    save(fig, '02-cay-vung-du-doan')


def boosting_figure():
    X, y = hand_regression(); model = SquaredBoost(2, .5, 1, 1).fit(X, y)
    stages = list(model.staged_predict(X)); grid = np.linspace(.6, 4.4, 400)[:, None]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    for i, color in [(0, BLUE), (1, ORANGE)]:
        residual = y-stages[i]
        axes[i].scatter(X[:, 0], residual, color=RED, s=55, label=f'r{i+1} = y − F{i}')
        axes[i].plot(grid[:, 0], model.trees[i].predict(grid), color=color, lw=2,
                     label=f'h{i+1}: stump fit residual')
        axes[i].axhline(0, color='#94a3b8', lw=1)
        axes[i].set(title=f'Vòng {i+1}: split {model.trees[i].root.threshold:g}',
                    xlabel='x', ylabel='Residual / dự đoán h', ylim=(-2, 3))
        axes[i].legend(loc='upper left', fontsize=9)
    axes[2].scatter(X[:, 0], y, color='black', s=45, label='y thật', zorder=4)
    grid_stages = list(model.staged_predict(grid))
    for i, color in [(0, '#64748b'), (1, BLUE), (2, GREEN)]:
        axes[2].plot(grid[:, 0], grid_stages[i], color=color, lw=2,
                     label=f'F{i}: train MSE={mse(y, stages[i]):.4f}')
    axes[2].set(title='Cộng 0.5×h sau mỗi vòng', xlabel='x', ylabel='Dự đoán tích lũy', ylim=(.4, 5.7))
    axes[2].legend(loc='upper left', fontsize=9)
    for ax in axes: ax.grid(alpha=.15)
    fig.suptitle('F₀=mean(y)=2.5; residual mới tính từ mô hình tích lũy hiện tại', fontsize=15, fontweight='bold')
    save(fig, '03-boosting-hai-vong')


def development_figures():
    train, val = development_data(); best, rows = select_model(train, val)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), layout='constrained')
    for ax, candidates, xs, label in [
            (axes[0], rows[2:5], [1, 2, 4], 'max_depth'),
            (axes[1], rows[5:], [1, 5, 20, 80, 200], 'Số vòng M (trục logarit)')]:
        ax.plot(xs, [r['train_mse'] for r in candidates], '-o', color=BLUE, label='Train MSE')
        ax.plot(xs, [r['val_mse'] for r in candidates], '-s', color=ORANGE, label='Validation MSE')
        ax.axhline(rows[1]['val_mse'], ls='--', color=GREEN, label='Linear validation MSE')
        ax.set(xlabel=label, ylabel='MSE (ms²)', title='Cây đơn: min_leaf=2' if ax is axes[0] else 'Boosting stump: η=0.1,min_leaf=2')
        ax.grid(alpha=.15); ax.legend(fontsize=9)
    axes[0].set_xticks([1, 2, 4]); axes[1].set_xscale('log')
    axes[1].set_xticks([1, 5, 20, 80, 200], labels=['1', '5', '20', '80', '200'])
    fig.suptitle('Train giảm chưa bảo đảm validation giảm; chỉ chọn bằng development', fontsize=15, fontweight='bold')
    save(fig, '04-train-validation')

    # The final test is accessed only after select_model has frozen its choice.
    tx, ty = final_test_data()
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.5), layout='constrained')
    names = [r['name'] for r in rows]; values = [r['val_mse'] for r in rows]
    bars = axes[0].barh(names, values, color=[GREEN if r is best else BLUE for r in rows])
    axes[0].bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set(title='Chọn bằng validation MSE', xlabel='MSE (ms²)', xlim=(0, 110))
    axes[0].grid(axis='x', alpha=.15)
    grid = np.linspace(-.2, 10.2, 300)[:, None]
    axes[1].scatter(train[0][:, 0], train[1], s=23, color=BLUE, alpha=.6, label='Train')
    axes[1].scatter(val[0][:, 0], val[1], s=25, marker='s', color=ORANGE, alpha=.6, label='Validation')
    axes[1].scatter(tx[:, 0], ty, s=45, marker='D', color=RED, label='Test sau chốt')
    axes[1].plot(grid[:, 0], model_predict(best['model'], grid), color=GREEN, lw=2.5,
                 label=f"Được chọn: {best['name']}")
    axes[1].axhline(rows[0]['model']['value'], color='#64748b', ls='--', label='Baseline mean train')
    axes[1].set(title='Dự đoán sau khi chốt lựa chọn', xlabel='Tải x (đơn vị giả định)', ylabel='Latency giả định (ms)')
    axes[1].legend(fontsize=9, loc='upper left'); axes[1].grid(alpha=.15)
    fig.suptitle('Baseline tuyến tính thắng trong lưới này; dữ liệu synthetic chưa là benchmark thật',
                 fontsize=14, fontweight='bold')
    save(fig, '05-baseline-so-sanh')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    gini_figure(); regions_figure(); boosting_figure(); development_figures()
    print(f'Generated 5 PNG + 5 SVG figures in {OUT}')


if __name__ == '__main__':
    main()
