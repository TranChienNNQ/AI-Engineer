"""Five original figures built from lab formulas and actual temporary study runs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from lab import optimizer_trace, decay_demo, quadratic_trace, failure_traces, study
OUT=Path(__file__).resolve().parent/'figures'
COLORS=['#2563eb','#ea580c','#15803d','#9333ea']

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150)
    fig.savefig(OUT/(name+'.svg'))
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
    for (name,rows),color in zip(optimizer_trace().items(),COLORS):
        ax.plot([0,1,2],[1]+[r['weight'] for r in rows],'-o',label=name,color=color)
    ax.set(xticks=[0,1,2],xlabel='Step',ylabel='Parameter θ',title='Gradient cho trước: 2 rồi 1; lr=0.1')
    ax.legend();ax.grid(alpha=.2);save(fig,'01-optimizers')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for lr,color in zip([.01,.1,.2,.25],COLORS):
        rows=quadratic_trace(lr)
        axes[0].plot([r['step'] for r in rows],[r['w'] for r in rows],'-o',ms=3,label=f'lr={lr}',color=color)
        axes[1].plot([r['step'] for r in rows],[r['loss'] for r in rows],'-o',ms=3,label=f'lr={lr}',color=color)
    axes[0].set_yscale('symlog',linthresh=1)
    axes[0].set(xlabel='Step',ylabel='w (symlog)',title='Đổi dấu và biên độ')
    axes[1].set_yscale('symlog',linthresh=.1)
    axes[1].set(xlabel='Step',ylabel='J(w) (symlog)',title='J(w)=5w², w₀=1')
    for ax in axes:ax.legend(fontsize=9);ax.grid(alpha=.2)
    save(fig,'02-learning-rate')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    w=np.linspace(-.5,4,250)
    for strength,color in zip([0,1,3],COLORS):
        axes[0].plot(w,.5*(w-3)**2+strength/2*w**2,label=f'λ={strength}',color=color)
        optimum=3/(1+strength)
        axes[0].scatter([optimum],[.5*(optimum-3)**2+strength/2*optimum**2],color=color,zorder=3)
    axes[0].set(xlabel='w',ylabel='Objective',title='L2 đổi mục tiêu và nghiệm');axes[0].legend()
    values=decay_demo();names=['SGD decay','Adam coupled','AdamW']
    bars=axes[1].bar(names,[values[k] for k in ['sgd_decay','adam_l2','adamw']],color=COLORS[:3])
    axes[1].set(ylim=(0,2.2),ylabel='w sau một bước',title='w₀=2, g_data=0, lr=0.1, decay=0.1')
    axes[1].bar_label(bars,fmt='%.2f',padding=3);save(fig,'03-regularization')
    with TemporaryDirectory(prefix='lesson13-figures-') as directory:
        root=Path(directory)/'study';report=study(root)
        history={r['run']:list(csv.DictReader((root/r['run']/'metrics.csv').open())) for r in report['candidates']}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name,color in [('plain',COLORS[0]),('dropout',COLORS[3])]:
        rows=history[name];epochs=[int(r['epoch']) for r in rows]
        axes[0].plot(epochs,[float(r['train_mse']) for r in rows],color=color,ls='--',label=name+' train')
        axes[0].plot(epochs,[float(r['val_mse']) for r in rows],color=color,label=name+' val')
    axes[0].set(xlabel='Epoch',ylabel='Data MSE, eval',title='Cuối run khác checkpoint best',ylim=(.05,.45));axes[0].legend(fontsize=9)
    for (name,rows),color in zip(history.items(),COLORS):
        axes[1].plot([int(r['epoch']) for r in rows],[float(r['weight_norm']) for r in rows],label=name,color=color)
    axes[1].set(xlabel='Epoch',ylabel='Norm weight, không bias',title='Norm nhỏ không tự bảo đảm val tốt nhất');axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'04-study')
    traces=failure_traces()
    fig,axes=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
    names=['Quên step','lr=0','ReLU chết']
    for ax,key,title in zip(axes,['loss','grad','update'],['Loss giống nhau','Gradient giúp khoanh vùng','Cần kiểm lr và step để phân biệt']):
        vals=[rows[0][key] for rows in traces.values()]
        bars=ax.bar(names,vals,color=COLORS[:3]);ax.bar_label(bars,padding=3,fmt='%g')
        ax.set(title=title,ylabel=key);ax.axhline(0,color='gray',lw=.8)
        ax.set_ylim((-.2,1.4) if key=='loss' else (-2.5,.5) if key=='grad' else (-.2,.2))
        ax.tick_params(axis='x',labelsize=10)
    fig.suptitle('Ba tình huống dựng có chủ đích, cùng loss phẳng ở 1')
    save(fig,'05-diagnosis');print('Created 5 PNG + 5 SVG')

if __name__=='__main__':main()
