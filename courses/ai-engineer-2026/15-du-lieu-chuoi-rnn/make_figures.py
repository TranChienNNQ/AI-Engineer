"""Five original sequence/RNN figures; runs live in a temporary directory."""
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from lab import make_records,hand_demo,padding_demo,train
OUT=Path(__file__).resolve().parent/'figures'
BLUE='#2563eb';ORANGE='#ea580c';GREEN='#15803d'

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150);fig.savefig(OUT/(name+'.svg'));plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    pair=make_records()[:2]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,row,color in zip(axes,pair,[BLUE,ORANGE]):
        values=row['values'];ax.plot(range(1,len(values)+1),values,'-o',color=color)
        ax.axhline(np.mean(values),color='gray',ls='--',label=f'Mean={np.mean(values):.4f}')
        ax.set(title='Nhãn 0: tăng' if row['label']==0 else 'Nhãn 1: giảm',xlabel='Bước t',ylabel='Feature xₜ');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Cùng group, cùng tập giá trị, đảo thứ tự → nhãn khác');save(fig,'01-sequences')
    hand=hand_demo()
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained');ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    xs=[.10,.35,.60,.85]
    for i,pos in enumerate(xs):
        text='h₀=0' if i==0 else f"h{i}={hand['states'][i-1]:.6f}\ntanh(0.5x + 0.25h)"
        ax.text(pos,.55,text,ha='center',va='center',fontsize=10,bbox={'boxstyle':'round,pad=.7','facecolor':'#dbeafe'})
        if i:
            ax.annotate('',xy=(pos-.09,.55),xytext=(xs[i-1]+.08,.55),arrowprops={'arrowstyle':'->','lw':2})
            ax.text(pos,.17,f"x{i}={hand['values'][i-1]:g}",ha='center')
            ax.annotate('',xy=(pos,.40),xytext=(pos,.23),arrowprops={'arrowstyle':'->'})
    ax.text(.5,.90,'Một bộ parameter dùng chung; hₜ đổi theo input và hₜ₋₁',ha='center',color=BLUE)
    ax.set_title('Unroll ba bước: cùng cell, ba lần tính');save(fig,'02-unrolled-rnn')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    steps=np.arange(21)
    for a,color in [(.5,BLUE),(1.,GREEN),(1.5,ORANGE)]:axes[0].semilogy(steps,a**steps,label=f'a={a}',color=color)
    axes[0].set(title='Toy tuyến tính hₜ=a hₜ₋₁',xlabel='Số bước T',ylabel='|∂h_T/∂h₀| = |a|^T');axes[0].legend();axes[0].grid(alpha=.2)
    bars=axes[1].bar([1,2,3],hand['deltas'],color=[BLUE,BLUE,ORANGE])
    axes[1].bar_label(bars,fmt='%.4f',padding=3);axes[1].set(title='BPTT ví dụ tanh: loss chỉ tại h₃',xlabel='Bước t',ylabel='δₜ=∂L/∂aₜ',xticks=[1,2,3],ylim=(0,.125))
    save(fig,'03-gradients')
    pad=padding_demo();fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    values=np.array(pad['padded']);axes[0].imshow(values,cmap='Blues',vmin=0,vmax=4)
    for i in range(2):
        for j in range(3):
            axes[0].text(j,i,f'{values[i,j]:g}',ha='center',va='center',color='white' if values[i,j]>2 else 'black')
            if j>=pad['lengths'][i]:axes[0].add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch='//',edgecolor=ORANGE,lw=2))
    axes[0].set(xticks=[0,1,2],xticklabels=['t1','t2','t3'],yticks=[0,1],yticklabels=['A: length1','B: length3'],title='Right-padding: vùng gạch là đệm')
    position=np.arange(2)
    axes[1].bar(position-.18,pad['naive_last'],width=.35,label='Naive cuối tensor',color=ORANGE)
    axes[1].bar(position+.18,pad['packed_last'],width=.35,label='Packed cuối thật',color=BLUE)
    axes[1].set(xticks=position,xticklabels=['Chuỗi A','Chuỗi B'],ylabel='Hidden state',title='ReLU cell: padding vẫn đổi state');axes[1].legend(fontsize=9)
    save(fig,'04-padding')
    with TemporaryDirectory(prefix='lesson15-figures-') as directory:
        root=Path(directory)/'run';report=train(root)
        rows=list(csv.DictReader((root/'metrics.csv').open()))
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained');epochs=[int(r['epoch']) for r in rows]
    for key,label in [('train_loss','Train eval'),('val_loss','Validation')]:axes[0].semilogy(epochs,[float(r[key]) for r in rows],label=label)
    axes[0].set(title='CE sau mỗi epoch',xlabel='Epoch',ylabel='CE (log)');axes[0].legend(fontsize=9)
    axes[1].plot(epochs,[float(r['val_accuracy']) for r in rows],label='RNN validation',color=BLUE)
    axes[1].axhline(.5,ls='--',label='Majority = 50%',color='gray')
    axes[1].axhline(report['baseline_val_accuracy']['endpoint_rule'],ls=':',label='Endpoint rule = 100%',color=GREEN)
    axes[1].set(title='Baseline đơn giản cũng giải được',xlabel='Epoch',ylabel='Accuracy',ylim=(0,1.05));axes[1].legend(fontsize=8)
    matrix=np.array(report['test']['confusion']);axes[2].imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2):axes[2].text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>8 else 'black')
    axes[2].set(xticks=[0,1],xticklabels=['Tăng','Giảm'],yticks=[0,1],yticklabels=['Tăng','Giảm'],xlabel='Predicted',ylabel='True',title='Test sau chọn best epoch')
    fig.suptitle(f"RNN 106 parameter; best epoch {report['best_epoch']}; test32 chuỗi /16 group")
    save(fig,'05-training');print('Created 5 PNG + 5 SVG')

if __name__=='__main__':main()
