"""Rebuild lesson 7 figures from the formulas and examples in lab.py."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lab import (choose_threshold, multiclass_counts, multiclass_report,
                 prevalence_precision, ranking_curves, ranking_example,
                 validation_scores)

OUT = Path(__file__).resolve().parent/'figures'
BLUE, ORANGE, GREEN, RED = '#2563eb', '#d97706', '#15803d', '#dc2626'


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=170)
    fig.savefig(OUT/f'{name}.svg')
    plt.close(fig)


def matrix_panel(ax, values, labels, title, percent=False, cell_names=None):
    ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if percent else values.max())
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel='Nhãn dự đoán', ylabel='Nhãn thật', title=title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = values[i, j]
            name = '' if cell_names is None else cell_names[i][j]+'\n'
            text = f'{value:.1%}' if percent else f'{value:g}'
            color = 'white' if value > .55*(1 if percent else values.max()) else '#0f172a'
            ax.text(j, i, name+text, ha='center', va='center', color=color, fontsize=13)


def confusion_figure():
    cm = np.array([[950, 30], [5, 15]])
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5), layout='constrained')
    matrix_panel(axes[0], cm, ['0', '1'], 'Bảng đếm: N=1000',
                 cell_names=[['TN', 'FP'], ['FN', 'TP']])
    matrix_panel(axes[1], cm/cm.sum(axis=1, keepdims=True), ['0', '1'],
                 'Theo hàng: trong mỗi lớp thật', percent=True)
    matrix_panel(axes[2], cm/cm.sum(axis=0, keepdims=True), ['0', '1'],
                 'Theo cột: trong mỗi dự đoán', percent=True)
    fig.suptitle('Cùng TP=15: recall = 15/20, precision = 15/45', fontsize=16, fontweight='bold')
    save(fig, '01-confusion-matrix')


def threshold_figure():
    y, scores = validation_scores()
    best, rows = choose_threshold(y, scores)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout='constrained')
    for label, color, marker in [(0, BLUE, 'o'), (1, ORANGE, 'D')]:
        mask = y == label
        axes[0].scatter(scores[mask], np.flatnonzero(mask)+1, color=color, marker=marker,
                        s=48, label=f'Nhãn thật {label}', zorder=3)
    for threshold, color in [(.7, RED), (.5, '#64748b'), (.4, GREEN)]:
        axes[0].axvline(threshold, ls='--', color=color, lw=1.4)
    axes[0].set(title='12 mẫu validation dựng sẵn', xlabel='Score (≥ ngưỡng → 1)',
                ylabel='Số thứ tự mẫu', xlim=(0, 1), yticks=range(1, 13))
    axes[0].legend(fontsize=9, loc='lower left')
    rows = sorted(rows, key=lambda row: row['threshold'])
    thresholds = [row['threshold'] for row in rows]
    for key, label, color, marker in [('precision', 'Precision', BLUE, 'o'),
                                      ('recall', 'Recall', ORANGE, 's'),
                                      ('f1', 'F1', GREEN, '^')]:
        axes[1].plot(thresholds, [row[key] for row in rows], color=color, marker=marker,
                     lw=2, label=label)
    axes[1].set(title='Accuracy cả ba = 2/3', xlabel='Ngưỡng τ', ylabel='Metric',
                xticks=thresholds, ylim=(0, 1.08))
    axes[1].legend(fontsize=10, loc='lower left'); axes[1].grid(alpha=.18)
    costs = [row['cost'] for row in rows]
    bars = axes[2].bar([f'{t:g}' for t in thresholds], costs, width=.55,
                       color=[GREEN if t == best['threshold'] else ORANGE for t in thresholds])
    axes[2].bar_label(bars, padding=5)
    axes[2].set(title='Chi phí = FP + 5×FN', xlabel='Ngưỡng τ', ylabel='Tổng chi phí validation',
                ylim=(0, 14))
    fig.suptitle('Chọn trong ba ngưỡng bằng validation cost; test ở ngoài quá trình chọn',
                 fontsize=15, fontweight='bold')
    save(fig, '02-nguong-metric-chi-phi')


def curves_figure():
    curves = ranking_curves(*ranking_example())
    r, p, f = curves['recall'], curves['precision'], curves['fpr']
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.2), layout='constrained')
    axes[0].plot(f, r, '-o', color=BLUE, lw=2, label='Các ngưỡng score khác nhau')
    axes[0].plot([0, 1], [0, 1], '--', color='#94a3b8', lw=1.2)
    axes[0].fill_between(f, r, alpha=.10, color=BLUE)
    axes[0].set(title=f"ROC-AUC = {curves['roc_auc']:.3f}", xlabel='FPR = FP / số negative thật',
                ylabel='TPR = recall', xlim=(-.04, 1.04), ylim=(-.04, 1.06))
    axes[0].annotate('τ=0.8: hai mẫu hòa vào cùng lúc', xy=(.5, 1), xytext=(.12, .76),
                     arrowprops={'arrowstyle': '->'}, fontsize=10)
    axes[1].step(r, p, where='pre', color=GREEN, lw=2, label='Các dải tính AP')
    axes[1].fill_between(r, p, step='pre', alpha=.18, color=GREEN)
    axes[1].plot(r, p, ':', color=ORANGE, lw=2, label='Nối thẳng: hình thang PR khác AP')
    axes[1].scatter(r[1:], p[1:], color=GREEN, s=45, zorder=4)
    axes[1].scatter([0], [1], marker='*', color=RED, s=100, zorder=5)
    axes[1].annotate('Điểm quy ước, không đo precision', xy=(0, 1), xytext=(.07, .83),
                     arrowprops={'arrowstyle': '->'}, fontsize=10)
    axes[1].set(title=f"AP = 5/6 ≈ {curves['ap']:.3f}; hình thang PR = 11/12",
                xlabel='Recall', ylabel='Precision', xlim=(-.04, 1.04), ylim=(-.04, 1.06))
    axes[1].legend(loc='lower left', fontsize=9)
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle('Bốn mẫu: y=[1,0,1,0], score=[0.9,0.8,0.8,0.1]', fontsize=16, fontweight='bold')
    save(fig, '03-roc-pr-ap')


def prevalence_figure():
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5), layout='constrained')
    tp, fp = np.array([800, 80]), np.array([180, 198])
    names = ['Positive thật 10%', 'Positive thật 1%']
    bars = axes[0].bar(names, tp, color=GREEN, label='TP kỳ vọng', width=.55)
    axes[0].bar_label(bars, label_type='center', color='white')
    bars = axes[0].bar(names, fp, bottom=tp, color=ORANGE, label='FP kỳ vọng', width=.55)
    axes[0].bar_label(bars, label_type='center', color='white')
    axes[0].set(title='Cảnh báo trong mỗi 10000 mẫu', ylabel='Số dự đoán positive kỳ vọng', ylim=(0, 1250))
    axes[0].legend(loc='upper right', fontsize=10)
    pi = np.geomspace(.001, .5, 400)
    axes[1].semilogx(100*pi, prevalence_precision(pi), color=BLUE, lw=2)
    for value in [.01, .1]:
        precision = float(prevalence_precision(value))
        axes[1].scatter([value*100], [precision], color=RED, s=50, zorder=3)
        axes[1].annotate(f'π={value:.0%}: P≈{precision:.1%}',
                         xy=(value*100, precision), xytext=(8, -20),
                         textcoords='offset points', fontsize=10)
    axes[1].set(title='Giữ TPR=80%, FPR=2%', xlabel='Tỷ lệ positive thật (%) — trục logarit',
                ylabel='Precision theo công thức xác suất', ylim=(0, 1.03))
    axes[1].grid(alpha=.18)
    fig.suptitle('Precision phụ thuộc tỷ lệ lớp, ngay cả khi TPR/FPR giữ nguyên',
                 fontsize=15, fontweight='bold')
    save(fig, '04-ty-le-lop-precision')


def multiclass_figure():
    cm = multiclass_counts(); report = multiclass_report(cm)
    names = ['billing', 'technical', 'account']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    matrix_panel(axes[0], cm/cm.sum(axis=1, keepdims=True), names,
                 'Chuẩn hóa theo hàng: support 90, 8, 2', percent=True)
    labels = names+['Macro-F1', 'Weighted-F1', 'Micro-F1']
    values = [*report['f1'], report['macro_f1'], report['weighted_f1'], report['micro_f1']]
    bars = axes[1].barh(labels, values, color=[BLUE, BLUE, RED, ORANGE, GREEN, '#64748b'])
    axes[1].bar_label(bars, labels=[f'{value:.3f}' for value in values], padding=4, fontsize=10)
    axes[1].invert_yaxis()
    axes[1].set(title='F1 từng lớp và các cách gộp', xlabel='F1', xlim=(0, 1.15))
    axes[1].grid(axis='x', alpha=.18)
    fig.suptitle('Accuracy=90% vẫn đi cùng recall và F1 của account bằng 0',
                 fontsize=16, fontweight='bold')
    save(fig, '05-da-lop-trung-binh')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    confusion_figure(); threshold_figure(); curves_figure(); prevalence_figure(); multiclass_figure()
    print(f'Generated 5 PNG + 5 SVG figures in {OUT}')


if __name__ == '__main__':
    main()
