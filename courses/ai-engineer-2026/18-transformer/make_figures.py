"""Rebuild seven original educational figures, PNG + SVG. No network needed."""
from pathlib import Path
import argparse
import csv
import json
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from lab import Config, train, sinusoidal

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
BLUE, TEAL, ORANGE, GRAY = '#2563eb', '#0f766e', '#c2410c', '#475569'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.fonttype': 'none'})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{extension}', dpi=160, bbox_inches='tight')
    plt.close(fig)


def box(ax, x, y, text, width=.76, height=.095, color=BLUE):
    ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                              boxstyle='round,pad=0.008', fc=color+'12', ec=color, lw=1.3))
    ax.text(x, y, text, ha='center', va='center', fontsize=10)


def arrow(ax, x1, y1, x2, y2, **kw):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.3, **kw))


def architectures():
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.3))
    titles = ['Encoder-only', 'Decoder-only (lab train)', 'Encoder–decoder']
    texts = [
        ['ID + vị trí', 'Self-attention hai chiều', 'Residual / LN / FFN', 'Stack → hidden từng token', 'Pooling/head theo tác vụ'],
        ['Prefix ID + vị trí', 'Causal self-attention', 'Residual / LN / FFN', 'Stack → final LN → head', 'Token tiếp theo → nối prefix'],
        ['Source → encoder → memory', 'Target đã shift + vị trí', 'Causal self-attention', 'Cross-attention: Q từ target\nK,V từ memory source', 'FFN / residual / LN → head']]
    for ax, title, lines in zip(axes, titles, texts):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
        ax.set_title(title, weight='bold', pad=15)
        ys = [.89, .72, .55, .38, .21]
        for i, (y, text) in enumerate(zip(ys, lines)):
            box(ax, .5, y, text, width=.92, height=.11)
            if i and not (title == 'Encoder–decoder' and i == 1):
                arrow(ax, .5, ys[i-1]-.06, .5, y+.06)
        if title == 'Encoder–decoder':
            # Memory feeds cross-attention on a separate branch, not the target embedding.
            ax.plot([.04, .01, .01], [.89, .89, .38], color=TEAL, lw=1.8)
            arrow(ax, .01, .38, .04, .38)
    fig.text(.5, .035, 'Nguồn QKV và vùng được đọc quyết định kiến trúc; không chỉ là tên module.', ha='center', color=GRAY)
    save(fig, '01-architectures')


def blocks():
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    for ax, pre in zip(axes, [False, True]):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
        ax.set_title('Pre-norm: chuẩn hóa trong nhánh' if pre else 'Post-norm: chuẩn hóa sau tổng', weight='bold')
        lines = ['X', 'LN₁ → MHA', 'U = X + output nhánh', 'LN₂ → FFN', 'Y = U + output nhánh'] if pre else [
            'X', 'MHA', 'U = LN₁(X + MHA(X))', 'FFN', 'Y = LN₂(U + FFN(U))']
        ys = [.90, .73, .54, .35, .16]
        for i, (y, text) in enumerate(zip(ys, lines)):
            box(ax, .52, y, text, width=.78, height=.10, color=TEAL if pre else BLUE)
            if i:
                arrow(ax, .52, ys[i-1]-.055, .52, y+.055)
        # Draw residual bypass paths outside boxes with labels, using right/left margins.
        ax.plot([.13, .055, .055, .13], [.90, .90, .54, .54], color=ORANGE, lw=1.7)
        ax.text(.04, .72, 'X', color=ORANGE, ha='right')
        ax.plot([.91, .98, .98, .91], [.54, .54, .16, .16], color=ORANGE, lw=1.7)
        ax.text(.985, .35, 'U', color=ORANGE, ha='left')
    fig.text(.5, .025, 'Lab tính tay: post-norm. Lab huấn luyện: pre-norm, thêm final LN sau stack. Dropout = 0.', ha='center', color=GRAY)
    save(fig, '02-blocks')


def hand():
    arrays = [np.array([[1,3],[2,0]]), np.full((2,2),.5), np.full((2,2),1.5),
              np.array([[2.5,4.5],[3.5,1.5]]), np.array([[-1,1],[1,-1]]),
              np.array([[0,2,1],[1,0,0]]), np.array([[1,1],[1,0]]), np.array([[-1,1],[1,-1]])]
    names = ['X: input', 'A: attention weight', 'Z = AX', 'R = X + Z', 'U = LN₁(R)',
             'ReLU(UW₁)', 'FFN(U)', 'Y = LN₂(U + FFN(U))']
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, array, title in zip(axes.flat, arrays, names):
        ax.imshow(array, cmap='coolwarm', vmin=-4.5, vmax=4.5)
        for (i,j), value in np.ndenumerate(array):
            ax.text(j,i,f'{value:g}',ha='center',va='center',fontsize=14)
        ax.set_title(title, fontsize=11)
        prefix = 'key ' if title.startswith('A:') else 'f'
        ax.set_xticks(range(array.shape[1]), [f'{prefix}{i}' for i in range(array.shape[1])])
        ax.set_yticks([0,1], ['token 0','token 1'])
    fig.suptitle('Tính tay một post-norm block: Q = K = 0, V = X, ε = 0 chỉ trong ví dụ', weight='bold')
    fig.tight_layout(rect=(0,0,1,.94), h_pad=2)
    save(fig, '03-hand')


def position():
    pe = sinusoidal(32,16).numpy()
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    im = axes[0].imshow(pe,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1)
    axes[0].set(title='PE: 32 vị trí × 16 chiều',xlabel='Chiều feature',ylabel='Vị trí token (bắt đầu 0)')
    fig.colorbar(im,ax=axes[0],fraction=.046)
    for i in range(4):
        axes[1].plot(np.arange(32),pe[:,i],label=f'chiều {i}')
    axes[1].set(title='Hai cặp sin/cos có tần số khác nhau',xlabel='Vị trí token',ylabel='Giá trị PE',ylim=(-1.15,1.15))
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=.2)
    fig.suptitle('Cộng PE vào embedding: giữ nguyên chiều D, không thêm token',weight='bold')
    fig.tight_layout(rect=(0,0,1,.93))
    save(fig,'04-position')


def shift_masks():
    labels = ['BOS','A','C','B','SEP','A','C','B']
    fig,axes=plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios':[1.15,1]})
    mask=np.triu(np.ones((8,8)),1)
    axes[0].imshow(mask,cmap='Blues',vmin=0,vmax=1)
    for i in range(8):
        for j in range(8):
            axes[0].text(j,i,'×' if mask[i,j] else '✓',ha='center',va='center',color='white' if mask[i,j] else TEAL)
    axes[0].set_xticks(range(8),labels,rotation=45)
    axes[0].set_yticks(range(8),labels)
    axes[0].set(xlabel='Key input',ylabel='Query input',title='Causal: xanh = chặn, ✓ = đọc')
    axes[1].axis('off')
    axes[1].text(0, .92, 'Một mẫu source [A,C,B]',fontsize=14,weight='bold')
    axes[1].text(0,.77,'Tại SEP: dùng prefix để đoán A.\nTại B cuối: đoán EOS.',linespacing=1.7)
    table=axes[1].table(cellText=[labels,['·','·','·','·','A','C','B','EOS'],['0','0','0','0','1','1','1','1']],
                         rowLabels=['input','label','loss?'],cellLoc='center',bbox=[.08,.27,.91,.29])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    axes[1].text(0,.1,'· = IGNORE; loss mask không chặn attention.\nPAD mask theo batch, tách riêng causal mask.',color=GRAY,linespacing=1.6)
    fig.suptitle('Shift + causal + padding + loss: các nhiệm vụ riêng',weight='bold')
    fig.tight_layout(rect=(0,0,1,.94))
    save(fig,'05-shift-masks')


def cost():
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    axes[0].axis('off')
    axes[0].set(xlim=(0,1),ylim=(0,1))
    lines=['ID [B,T]', 'Embedding + PE [B,T,D]', 'Q,K,V mỗi head [B,H,T,D/H]', 'Score [B,H,T,T]', 'Output / residual / FFN [B,T,D]', 'Logits [B,T,V]']
    ys=np.linspace(.91,.12,len(lines))
    for i,(y,text) in enumerate(zip(ys,lines)):
        box(axes[0],.5,y,text,width=.96,height=.09)
        if i: arrow(axes[0],.5,ys[i-1]-.05,.5,y+.05)
    lengths=np.array([512,1024,2048])
    memory=1*4*lengths**2*4/2**20
    axes[1].bar([str(t) for t in lengths],memory,color=[TEAL,BLUE,ORANGE])
    for i,m in enumerate(memory): axes[1].text(i,m+1.5,f'{m:g} MiB',ha='center')
    axes[1].set(xlabel='T',ylabel='MiB',ylim=(0,75),title='Một tensor score float32: B=1, H=4')
    axes[1].text(.5,.8,'T gấp 2 → score gấp 4',transform=axes[1].transAxes,ha='center',color=GRAY)
    fig.suptitle('Shape và chi phí minh họa; không phải tổng RAM hoặc benchmark GPU',weight='bold')
    fig.tight_layout(rect=(0,0,1,.94))
    save(fig,'06-shapes-cost')


def training(run):
    report=json.loads((run/'report.json').read_text())
    with (run/'metrics.csv').open() as f: rows=list(csv.DictReader(f))
    epochs=[int(r['epoch']) for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    ax=axes[0,0]
    ax.plot(epochs,[float(r['train_ce']) for r in rows],label='Train: trong cập nhật')
    ax.plot(epochs,[float(r['val_ce']) for r in rows],label='Validation: cuối epoch')
    ax.axvline(report['best_epoch'],color=GRAY,ls='--',label=f"Best epoch {report['best_epoch']}")
    ax.set(title='Cross-entropy trên token copy + EOS',xlabel='Epoch',ylabel='CE/token')
    ax.legend(fontsize=9)
    ax=axes[0,1]
    ax.plot(epochs,[100*float(r['val_token_accuracy']) for r in rows],color=TEAL)
    ax.set(title='Validation token accuracy, teacher forcing',xlabel='Epoch',ylabel='%',ylim=(0,102))
    ax=axes[1,0]
    names=['Token\n(teacher forcing)','Cả chuỗi\n(greedy)']
    x=np.arange(2)
    for offset,split,color in [(-.18,'validation',TEAL),(.18,'test',BLUE)]:
        vals=[report[split]['token_accuracy']*100,report[split]['exact_match']*100]
        bars=ax.bar(x+offset,vals,.36,label=split,color=color)
        ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
    ax.set_xticks(x,names)
    ax.set(title='Hai điều kiện đánh giá khác nhau',ylabel='%',ylim=(0,115))
    ax.legend(fontsize=9)
    ax=axes[1,1]
    vals=[report['test']['exact_match']*100,100.,0.]
    bars=ax.bar(['Transformer','Quy tắc copy','Luôn sinh B'],vals,color=[BLUE,TEAL,ORANGE])
    ax.bar_label(bars,fmt='%.1f%%',padding=3)
    ax.set(title='Test exact match, cùng source đã cho',ylabel='%',ylim=(0,115))
    for ax in axes.flat: ax.grid(axis='y',alpha=.2)
    fig.suptitle(f"Copy chuỗi nhân tạo: {report['parameters']:,} tham số, chọn checkpoint bằng validation CE",weight='bold')
    fig.tight_layout(rect=(0,0,1,.95),h_pad=2)
    save(fig,'07-training')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    architectures(); blocks(); hand(); position(); shift_masks(); cost()
    if args.run:
        training(args.run)
    else:
        with tempfile.TemporaryDirectory(prefix='lesson18-fig-') as directory:
            run=Path(directory)/'run'
            train(run,Config())
            training(run)
    print('Saved 7 PNG + 7 SVG:',OUT)

if __name__=='__main__':
    main()
