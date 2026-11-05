"""Rebuild five lesson 10 figures from the graph, gradients and trained MLP."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lab import (KEYS, forward, gradient_check, hand_example, loss_and_gradients,
                 scalar_example, sigmoid, train_xor, xor_data)

OUT = Path(__file__).resolve().parent/'figures'
BLUE, ORANGE, GREEN, RED = '#2563eb', '#d97706', '#15803d', '#dc2626'


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=170)
    fig.savefig(OUT/f'{name}.svg')
    plt.close(fig)


def box(ax, x, y, text, color='#dbeafe', size=11):
    ax.text(x, y, text, ha='center', va='center', fontsize=size, zorder=4,
            bbox={'boxstyle': 'round,pad=.5', 'facecolor': color, 'edgecolor': '#64748b'})


def arrow(ax, start, end, color='#64748b', label=None, shift=(0, 0), shrink=26):
    ax.annotate('', xy=end, xytext=start, zorder=1,
                arrowprops={'arrowstyle': '->', 'lw': 1.5, 'color': color,
                            'shrinkA': shrink, 'shrinkB': shrink})
    if label:
        x, y = (start[0]+end[0])/2+shift[0], (start[1]+end[1])/2+shift[1]
        ax.text(x, y, label, ha='center', va='center', fontsize=10, color=color,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1}, zorder=3)


def scalar_figure():
    values = scalar_example()
    fig, ax = plt.subplots(figsize=(13.2, 5), layout='constrained')
    nodes = {'x': (.05, .78), 'w': (.05, .38), 'u': (.25, .60), 'b': (.25, .18),
             'a': (.46, .60), 'v': (.69, .83), 'r': (.69, .35), 'L': (.93, .60)}
    for source, target in [('x','u'),('w','u'),('u','a'),('b','a'),('a','v'),('a','r'),('v','L'),('r','L')]:
        arrow(ax, nodes[source], nodes[target])
    labels = {'x':'x=2', 'w':'w=0.5', 'u':'u=wx=1', 'b':'b=1', 'a':'a=u+b=2',
              'v':'v=a²=4', 'r':'r=3a=6', 'L':'L=v+r=10'}
    for name, point in nodes.items():
        box(ax, *point, labels[name], color='#dcfce7' if name == 'L' else '#ffedd5' if name == 'a' else '#dbeafe')
    ax.text(.58, .88, f"Nhánh a² góp {values['da_square']:g}", color=RED, ha='center', fontsize=11)
    ax.text(.58, .28, f"Nhánh 3a góp {values['da_linear']:g}", color=RED, ha='center', fontsize=11)
    ax.text(.46, .47, 'dL/da = 4 + 3 = 7', color=RED, ha='center', fontsize=12,
            bbox={'facecolor': 'white', 'edgecolor': 'none'})
    ax.text(.5, .05, 'Kết quả backward: dL/dw=14, dL/db=7, dL/dx=3.5',
            ha='center', fontsize=13, color=GREEN)
    ax.set(xlim=(-.02, 1.04), ylim=(-.03, 1.02)); ax.axis('off')
    fig.suptitle('Đồ thị có nhánh: nhân dọc đường, cộng các đóng góp về cùng biến',
                 fontsize=16, fontweight='bold')
    save(fig, '01-graph-nhanh')


def shape_figure():
    fig, ax = plt.subplots(figsize=(14, 5.6), layout='constrained')
    xs = [.07, .29, .51, .73, .95]
    top = ['X\n(n,d)', 'Z₁=XW₁+b₁\n(n,h)', 'H=φ(Z₁)\n(n,h)', 'Z₂=HW₂+b₂\n(n,1)', 'Mean BCE\nscalar']
    bottom = ['dX=G₁W₁ᵀ\n(n,d)', 'G₁=G_H ⊙ φ′(Z₁)\n(n,h)',
              'G_H=G₂W₂ᵀ\n(n,h)', 'G₂=(P−Y)/n\n(n,1)', 'Seed\ndL/dL=1']
    for i, x in enumerate(xs):
        box(ax, x, .80, top[i], '#dbeafe', 11)
        box(ax, x, .42, bottom[i], '#fee2e2', 11)
        if i < 4:
            arrow(ax, (x, .80), (xs[i+1], .80), BLUE, shrink=52)
            arrow(ax, (xs[i+1], .42), (x, .42), RED, shrink=55)
    arrow(ax, (.95, .72), (.95, .50), RED, shrink=3)
    ax.text(.29, .12, 'dW₁ = XᵀG₁ + λW₁\ndb₁ = sum_rows(G₁)', ha='center', color=GREEN, fontsize=12)
    ax.text(.73, .12, 'dW₂ = HᵀG₂ + λW₂\ndb₂ = sum_rows(G₂)', ha='center', color=GREEN, fontsize=12)
    ax.text(.51, .97, 'Forward →       ← Backward', ha='center', fontsize=13)
    ax.set(xlim=(-.04, 1.06), ylim=(-.02, 1.08)); ax.axis('off')
    fig.suptitle('Một lần chia n ở đầu backward; mọi gradient dùng weight của cùng forward',
                 fontsize=16, fontweight='bold')
    save(fig, '02-forward-backward-shape')


def heatmap(ax, values, title, mask=False):
    if mask:
        ax.imshow(values, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    else:
        ax.imshow(values, cmap='coolwarm', vmin=-.4, vmax=.4, aspect='auto')
    ax.set(title=title, xticks=range(values.shape[1]), yticks=range(values.shape[0]),
           xticklabels=[f'cột {i+1}' for i in range(values.shape[1])],
           yticklabels=[f'hàng {i+1}' for i in range(values.shape[0])])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i,j]
            ax.text(j, i, f'{int(v)}' if mask else f'{v:.6f}', ha='center', va='center', fontsize=12,
                    color='white' if (mask and v == 1) or (not mask and abs(v) > .26) else '#0f172a')


def hand_figure():
    X, y, p = hand_example(); _, _, grads, cache = loss_and_gradients(X, y, p)
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.6), layout='constrained')
    items = [(cache['dZ2'], 'G₂=(P−Y)/2', False), (cache['dH'], 'G_H=G₂W₂ᵀ', False),
             ((cache['Z1']>0).astype(int), 'ReLU mask từ Z₁', True),
             (cache['dZ1'], 'G₁=G_H ⊙ mask', False), (grads['W1'], 'dW₁=XᵀG₁', False),
             (grads['W2'], 'dW₂=HᵀG₂', False)]
    for ax, (values, title, mask) in zip(axes.flat, items): heatmap(ax, values, title, mask)
    fig.suptitle('Tính tay batch hai mẫu: q=σ(−1.5)≈0.182425524; db₁=[−q,0], db₂=[0]',
                 fontsize=15, fontweight='bold')
    save(fig, '03-backprop-tinh-tay')


def check_figure():
    X, y, p = hand_example()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    positions = np.arange(4)
    for activation, offset, color in [('relu', -.18, BLUE), ('tanh', .18, ORANGE)]:
        rows = gradient_check(X, y, p, activation, .01)
        values = [max(row['max_abs'], 1e-18) for row in rows]
        axes[0].bar(positions+offset, values, width=.34, color=color, label=activation)
    axes[0].set(xticks=positions, xticklabels=KEYS, yscale='log', ylim=(1e-18, 1e-8),
                title='Sai số tuyệt đối lớn nhất theo tensor', ylabel='max |numeric − backprop| (trục log)')
    axes[0].legend(fontsize=10); axes[0].grid(axis='y', alpha=.15)
    epsilon = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
    errors = [max(row['max_abs'] for row in gradient_check(X,y,p,'tanh',.01,float(eps))) for eps in epsilon]
    axes[1].loglog(epsilon, errors, '-o', color=GREEN)
    axes[1].set(title='Tanh: ε quá nhỏ có thể tăng sai số', xlabel='Epsilon (trục log)',
                ylabel='max sai số tuyệt đối của mọi parameter')
    axes[1].grid(alpha=.15)
    fig.suptitle('Gradient check cùng objective có L2=0.01, float64; không suy từ relative một mình',
                 fontsize=14, fontweight='bold')
    save(fig, '04-gradient-check')


def train_figure():
    X, y = xor_data(); result = train_xor()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), layout='constrained')
    history = result['history']
    axes[0].semilogy(history[:,0],history[:,1],color=BLUE,lw=2,label='Train BCE')
    axes[0].semilogy(history[:,0],history[:,2],color=ORANGE,lw=2,ls='--',label='Objective có L2')
    axes[0].set(title='Cập nhật cả W₁,b₁,W₂,b₂', xlabel='Số bước', ylabel='Loss / objective, trục log')
    axes[0].legend(fontsize=10); axes[0].grid(alpha=.15)
    a,b=np.meshgrid(np.linspace(-.1,1.1,200),np.linspace(-.1,1.1,200))
    grid=np.column_stack([a.ravel(),b.ravel()])
    for ax, params, title in [(axes[1],result['initial'],'Khởi tạo ngẫu nhiên: seed=2026'),
                              (axes[2],result['params'],'Sau 5000 bước: accuracy train=1')]:
        probability=sigmoid(forward(grid,params,'tanh')[0][:,0]).reshape(a.shape)
        shaded=ax.contourf(a,b,probability,levels=np.linspace(0,1,21),cmap='viridis')
        if probability.min()<.5<probability.max():ax.contour(a,b,probability,levels=[.5],colors='white',linewidths=1.7)
        for label,color,marker in [(0,BLUE,'o'),(1,ORANGE,'D')]:
            ax.scatter(X[y==label,0],X[y==label,1],color=color,marker=marker,s=65,edgecolors='white',zorder=4,label=f'y={label}')
        ax.set(title=title,xlabel='x₁',ylabel='x₂',aspect='equal')
    axes[1].legend(loc='center right',fontsize=9)
    fig.colorbar(shaded,ax=axes[1:],label='p(Y=1) từ mạng',ticks=[0,.5,1],shrink=.8)
    fig.suptitle('XOR 2→4→1 tanh: bốn hàng train; vùng giữa các điểm chưa có nhãn tham chiếu',
                 fontsize=15,fontweight='bold')
    save(fig,'05-xor-end-to-end')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    scalar_figure(); shape_figure(); hand_figure(); check_figure(); train_figure()
    print(f'Generated 5 PNG + 5 SVG figures in {OUT}')


if __name__=='__main__':main()
