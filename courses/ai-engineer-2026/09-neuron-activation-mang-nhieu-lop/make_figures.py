"""Create five lesson 9 figures using the same arrays and formulas as lab.py."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lab import (activate, forward, hand_example, readout_experiment,
                 sigmoid, xor_data, xor_features, xor_layers)

OUT = Path(__file__).resolve().parent/'figures'
BLUE, ORANGE, GREEN, RED = '#2563eb', '#d97706', '#15803d', '#dc2626'


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=170)
    fig.savefig(OUT/f'{name}.svg')
    plt.close(fig)


def neuron_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), layout='constrained')
    ax = axes[0]
    boxes = [(0.5, .82, 'x=(2, −1), w=(0.5, −2), b=−0.5', '#dbeafe'),
             (.5, .52, 'z = 0.5×2 + (−2)×(−1) − 0.5 = 2.5', '#ffedd5'),
             (.5, .20, 'a = ReLU(z) = max(0, 2.5) = 2.5', '#dcfce7')]
    for x, y, text, color in boxes:
        ax.text(x, y, text, ha='center', va='center', fontsize=11,
                bbox={'boxstyle': 'round,pad=.7', 'facecolor': color, 'edgecolor': '#94a3b8'})
    for top, bottom, label in [(.75, .59, 'Affine'), (.45, .27, 'Activation')]:
        ax.annotate('', xy=(.5, bottom), xytext=(.5, top), arrowprops={'arrowstyle': '->', 'lw': 1.8})
        ax.text(.54, (top+bottom)/2, label, va='center', fontsize=10)
    ax.set_title('Một neuron: tổng có trọng số → activation')
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')

    ax = axes[1]
    inputs, hidden, output = [(.12, .73), (.12, .30)], [(.50, .73), (.50, .30)], (.88, .52)
    for source in inputs:
        for target in hidden:
            ax.annotate('', xy=target, xytext=source, arrowprops={'arrowstyle': '->', 'color': '#94a3b8'}, zorder=1)
    for source in hidden:
        ax.annotate('', xy=output, xytext=source, arrowprops={'arrowstyle': '->', 'color': '#94a3b8'}, zorder=1)
    nodes = [(inputs[0], 'x₁=1', '#dbeafe'), (inputs[1], 'x₂=2', '#dbeafe'),
             (hidden[0], 'h₁=1', '#ffedd5'), (hidden[1], 'h₂=1', '#ffedd5'), (output, 'z=1.5', '#dcfce7')]
    for (x, y), text, color in nodes:
        ax.text(x, y, text, ha='center', va='center', fontsize=11,
                bbox={'boxstyle': 'circle,pad=.65', 'facecolor': color, 'edgecolor': '#64748b'}, zorder=3)
    ax.text(.12, .96, 'Input', ha='center'); ax.text(.5, .96, 'Hidden ReLU', ha='center')
    ax.text(.88, .96, 'Output logit', ha='center')
    ax.text(.5, .08, 'Lớp 1: 2×2 + 2 = 6 tham số\nLớp 2: 2×1 + 1 = 3 tham số',
            ha='center', va='center', fontsize=11)
    ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.08)); ax.axis('off')
    ax.set_title('Ví dụ 2→2→1: một hidden, hai lớp affine')
    fig.suptitle('Neuron và mạng nhỏ: weight/bias khác activation của từng mẫu', fontsize=16, fontweight='bold')
    save(fig, '01-neuron-mang-nho')


def activation_figure():
    z = np.linspace(-4, 4, 500)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.1), layout='constrained')
    for ax, name, title, color in [(axes[0, 0], 'sigmoid', 'Sigmoid: miền (0,1), bão hòa hai phía', BLUE),
                                   (axes[0, 1], 'tanh', 'Tanh: miền (−1,1)', ORANGE)]:
        ax.plot(z, activate(z, name), color=color, lw=2.3)
        ax.set(title=title, xlabel='z', ylabel='Activation')
    axes[1, 0].plot(z, activate(z, 'relu'), color=BLUE, lw=2.3, label='ReLU')
    axes[1, 0].plot(z, activate(z, 'leaky_relu'), color=GREEN, ls='--', lw=2.3, label='Leaky ReLU, α=0.1')
    axes[1, 0].set(title='Tuyến tính từng đoạn, phi tuyến toàn cục', xlabel='z', ylabel='Activation')
    axes[1, 0].legend(fontsize=10)
    axes[1, 1].plot(z, activate(z, 'gelu'), color=RED, lw=2.3, label='GELU = z Φ(z)')
    axes[1, 1].plot(z, activate(z, 'relu'), color='#64748b', ls='--', lw=1.6, label='ReLU để đối chiếu')
    axes[1, 1].set(title='GELU không phải xác suất đầu ra', xlabel='z', ylabel='Activation')
    axes[1, 1].legend(fontsize=10)
    for ax in axes.flat:
        ax.axhline(0, color='#94a3b8', lw=.7); ax.axvline(0, color='#94a3b8', lw=.7); ax.grid(alpha=.15)
    fig.suptitle('Activation đổi giá trị, giữ nguyên shape; không phải hàm nào cũng cho xác suất',
                 fontsize=15, fontweight='bold')
    save(fig, '02-activation')


def forward_figure():
    X, layers = hand_example(); logits, trace = forward(X, layers)
    data = [(X, 'X (2,2)'), (trace[0]['Z'], 'Z¹ (2,2)'), (trace[0]['A'], 'H=ReLU(Z¹) (2,2)'),
            (logits, 'Z²: logit (2,1)'), (sigmoid(logits), 'p=sigmoid(Z²) (2,1)')]
    fig, axes = plt.subplots(1, 5, figsize=(14.2, 4.1), layout='constrained')
    for index, (ax, (values, title)) in enumerate(zip(axes, data)):
        probabilities = index == 4
        ax.imshow(values, cmap='YlGn' if probabilities else 'coolwarm',
                  vmin=0 if probabilities else -2, vmax=1 if probabilities else 2, aspect='auto')
        ax.set(title=title, xticks=range(values.shape[1]), yticks=[0, 1],
               xticklabels=[f'cột {j+1}' for j in range(values.shape[1])], yticklabels=['mẫu 1', 'mẫu 2'])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                ax.text(j, i, f'{value:.6f}' if probabilities else f'{value:g}', ha='center', va='center',
                        fontsize=13, color='white' if (not probabilities and abs(value) >= 1.5) or (probabilities and value > .7) else '#0f172a')
    fig.suptitle('Truyền xuôi: XW¹+b¹ → Z¹ → ReLU → H → HW²+b² → logit → xác suất',
                 fontsize=14, fontweight='bold')
    save(fig, '03-forward-batch')


def xor_figure():
    X, y = xor_data(); layers = xor_layers()
    logits, trace = forward(X, layers); H = trace[0]['A']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout='constrained')
    a, b = np.meshgrid(np.linspace(-.15, 1.15, 260), np.linspace(-.15, 1.15, 260))
    grid = np.column_stack([a.ravel(), b.ravel()])
    probability = sigmoid(forward(grid, layers)[0][:, 0]).reshape(a.shape)
    shaded = axes[0].contourf(a, b, probability, levels=np.linspace(0, 1, 21), cmap='viridis')
    axes[0].contour(a, b, probability, levels=[.5], colors=['white'], linewidths=2)
    for label, color, marker in [(0, BLUE, 'o'), (1, ORANGE, 'D')]:
        axes[0].scatter(X[y == label, 0], X[y == label, 1], s=90, color=color, marker=marker,
                        edgecolors='white', label=f'Nhãn {label}', zorder=4)
    axes[0].set(title='X gốc: biên |x₁−x₂|=0.5', xlabel='x₁', ylabel='x₂', aspect='equal')
    axes[0].legend(fontsize=9, loc='center left')
    fig.colorbar(shaded, ax=axes[0], label='p từ weight đặt tay', ticks=[0, .5, 1])
    axes[1].plot([0, .5], [.5, 0], color=RED, lw=2, label='h₁+h₂=0.5')
    axes[1].scatter([0], [0], color=BLUE, s=95, edgecolors='white', zorder=4)
    axes[1].scatter(H[y == 1, 0], H[y == 1, 1], color=ORANGE, marker='D', s=95,
                    edgecolors='white', zorder=4)
    for point, text, offset in [((0, 0), '(0,0) và (1,1): y=0', (10, 12)),
                                ((1, 0), '(1,0): y=1', (-85, 12)),
                                ((0, 1), '(0,1): y=1', (10, -20))]:
        axes[1].annotate(text, point, xytext=offset, textcoords='offset points', fontsize=10)
    axes[1].set(title='H: ba vị trí, phân tách bằng đường thẳng', xlabel='h₁=ReLU(x₁−x₂)',
                ylabel='h₂=ReLU(x₂−x₁)', xlim=(-.1, 1.15), ylim=(-.1, 1.15), aspect='equal')
    axes[1].legend(loc='upper right', fontsize=10); axes[1].grid(alpha=.15)
    fig.suptitle('Biểu diễn hidden có thể thay khả năng phân tách; ví dụ này chưa học weight',
                 fontsize=15, fontweight='bold')
    save(fig, '04-xor-bieu-dien')


def readout_figure():
    X, y = xor_data(); models = readout_experiment()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for name, label, color in [('raw', 'Raw X: train BCE', BLUE), ('fixed_relu', 'H cố định: train BCE', GREEN)]:
        history = models[name]['history']
        axes[0].plot(history[:, 0], history[:, 1], color=color, lw=2, label=label)
    history = models['fixed_relu']['history']
    axes[0].plot(history[:, 0], history[:, 2], color=ORANGE, ls='--', label='H cố định: objective có L2')
    axes[0].set(title='Chỉ fit head, không cập nhật W¹,b¹', xlabel='Số bước gradient head',
                ylabel='Loss / objective (nat)', ylim=(.1, .75))
    axes[0].legend(fontsize=9); axes[0].grid(alpha=.15)
    index = np.arange(4)
    for name, offset, color in [('raw', -.18, BLUE), ('fixed_relu', .18, GREEN)]:
        model = models[name]; H = X if name == 'raw' else xor_features(X)
        p = sigmoid(H@model['w']+model['b'])
        bars = axes[1].bar(index+offset, p, width=.34, color=color, label=name)
        axes[1].bar_label(bars, labels=[f'{v:.3f}' for v in p], padding=3, fontsize=9)
    axes[1].axhline(.5, color=RED, ls='--', label='Ngưỡng 0.5, hòa gán 1')
    axes[1].set(title='Xác suất head đã fit trên bốn hàng train', xticks=index,
                xticklabels=['00\ny=0', '10\ny=1', '01\ny=1', '11\ny=0'],
                ylabel='p(Y=1)', ylim=(0, 1.15))
    axes[1].legend(fontsize=9, loc='upper center', ncols=2)
    fig.suptitle('3000 bước, lr=0.2, L2=0.01; đây không phải điểm validation/test',
                 fontsize=15, fontweight='bold')
    save(fig, '05-hoc-head-co-dinh')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    neuron_figure(); activation_figure(); forward_figure(); xor_figure(); readout_figure()
    print(f'Generated 5 PNG + 5 SVG figures in {OUT}')


if __name__ == '__main__':
    main()
