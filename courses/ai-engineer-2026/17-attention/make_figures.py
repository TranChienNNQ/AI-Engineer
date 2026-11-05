"""Seven original figures from exact hand arrays, API checks and a temporary learned recall run."""
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from lab import hand_demo,matrix_demo,masks_demo,train
OUT=Path(__file__).resolve().parent/'figures'
BLUE='#2563eb';ORANGE='#ea580c';GREEN='#15803d'

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150);fig.savefig(OUT/(name+'.svg'));plt.close(fig)

def heat(ax,a,title,rows=None,cols=None,fmt='.3f',vmin=None,vmax=None,cmap='Blues'):
    a=np.asarray(a);ax.imshow(a,cmap=cmap,vmin=vmin,vmax=vmax);ax.set_title(title,fontsize=11)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j,i,format(a[i,j],fmt),ha='center',va='center',color='white' if a[i,j]>(a.min()+a.max())/2 else 'black',fontsize=10)
    ax.set_xticks(range(a.shape[1]),cols if cols else range(a.shape[1]))
    ax.set_yticks(range(a.shape[0]),rows if rows else range(a.shape[0]))


def main():
    OUT.mkdir(exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    hand=hand_demo();matrix=matrix_demo();masks=masks_demo()
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    axes[0].axis('off');axes[0].text(.5,.5,'q = ln2\nK = [0,1,2]ᵀ\n↓ dot / √1\nScores = [0,ln2,ln4]\n↓ softmax\nA = [1,2,4] / 7',ha='center',va='center',linespacing=1.6,bbox={'boxstyle':'round,pad=.7','facecolor':'#dbeafe'})
    axes[0].set_title('Query so với key')
    bars=axes[1].bar(['key1','key2','key3'],hand['weights'][0],color=[BLUE,BLUE,ORANGE])
    axes[1].bar_label(bars,labels=['1/7','2/7','4/7'],padding=3);axes[1].set(title='Trọng số qua các key',ylabel='Weight',ylim=(0,.7))
    axes[2].axis('off');axes[2].text(.5,.5,'V₁=[1,0], V₂=[0,2]\nV₃=[3,1]\n\nZ = (1V₁+2V₂+4V₃)/7\n= [13/7,8/7]\n≈ [1.8571,1.1429]',ha='center',va='center',linespacing=1.6,bbox={'boxstyle':'round,pad=.7','facecolor':'#dcfce7'})
    axes[2].set_title('Trộn value, không trộn key');fig.suptitle('Một query: score → softmax → weighted sum');save(fig,'01-qkv')
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    heat(axes[0],matrix['scores'],'QKᵀ / √2: (2,3)',['q1','q2'],['k1','k2','k3'])
    heat(axes[1],matrix['weights'],'A: softmax theo từng hàng',['q1','q2'],['k1','k2','k3'])
    heat(axes[2],matrix['output'],'Z=AV: (2,2)',['q1','q2'],['v-dim1','v-dim2'])
    fig.suptitle('Hai query, ba key/value; output giữ số query');save(fig,'02-matrix')
    scores=np.array([0.,np.log(2),np.log(4)])
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for i,tau in enumerate([.5,1.,2.]):
        ex=np.exp(scores/tau-np.max(scores/tau));p=ex/ex.sum()
        axes[0].bar(np.arange(3)+(i-1)*.24,p,width=.24,label=f'τ={tau}')
    axes[0].set(xticks=[0,1,2],xticklabels=['key1','key2','key3'],ylabel='Weight',title='Score / τ trước softmax');axes[0].legend()
    contrast=np.linspace(-8,8,200);p=1/(1+np.exp(-contrast))
    axes[1].plot(contrast,p,label='Weight key1 (2 key)');axes[1].plot(contrast,p*(1-p),label='Đạo hàm theo chênh score')
    axes[1].set(xlabel='s₁−s₂',title='Softmax quá sắc: gradient có thể nhỏ');axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
    save(fig,'03-softmax')
    fig,axes=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
    heat(axes[0],masks['causal_allowed'],'Causal: 1=được đọc',['q1','q2','q3'],['k1','k2','k3'],fmt='.0f',vmin=0,vmax=1)
    heat(axes[1],masks['causal_weights'][0],'A sau causal mask',['q1','q2','q3'],['k1','k2','k3'],vmin=0,vmax=1)
    padding=np.array([[1,1,0],[1,1,0],[1,1,0]])
    heat(axes[2],padding,'Key3 PAD: chặn cột, không xóa query3',['q1','q2','q3'],['k1','k2','PAD'],fmt='.0f',vmin=0,vmax=1)
    fig.suptitle('Mask là hợp đồng đường đọc; không suy output query PAD bằng 0');save(fig,'04-masks')
    fig,ax=plt.subplots(figsize=(12,4.6),layout='constrained');ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    nodes=[(.1,'X\nB,T,D\n2,3,4'),(.32,'Q/K/V Linear\n→ B,T,4\nsplit + transpose'),(.57,'2 head\nQ: 2,2,3,2\nA: 2,2,3,3'),(.81,'Concat\n→ B,T,4\nW_O → B,T,4')]
    for x,t in nodes:ax.text(x,.59,t,ha='center',va='center',linespacing=1.5,bbox={'boxstyle':'round,pad=.7','facecolor':'#dbeafe'},fontsize=11)
    for start,end in [(.17,.22),(.43,.47),(.67,.72)]:ax.annotate('',xy=(end,.59),xytext=(start,.59),arrowprops={'arrowstyle':'->','lw':2})
    ax.text(.5,.17,'Mỗi head softmax riêng trên key; concat các head của cùng vị trí\nD=4,h=2,d_k=d_v=2; QKV + output projection = 80 parameter có bias',ha='center',linespacing=1.7)
    ax.set_title('Multi-head: cùng input, nhiều không gian đọc');save(fig,'05-multihead')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    bars=axes[0].bar(['s1','s2','s3'],hand['d_scores'][0],color=[BLUE,BLUE,ORANGE]);axes[0].bar_label(bars,fmt='%.4f',padding=3)
    axes[0].axhline(0,color='gray');axes[0].set(title='Loss=z₁: gradient score cộng bằng 0',ylabel='∂L/∂s',ylim=(-.68,.82))
    axes[1].axis('off');axes[1].text(.5,.55,'V₁=V₂=V₃=[1,2]\n\nA=[1/3,1/3,1/3]\nA′=[.99,.005,.005]\n\nCả hai cho Z=[1,2]',ha='center',va='center',linespacing=1.6,bbox={'boxstyle':'round,pad=.8','facecolor':'#ffedd5'})
    axes[1].set_title('Weight khác chưa đủ suy output khác');save(fig,'06-gradient-interpretation')
    with TemporaryDirectory(prefix='lesson17-figures-') as directory:
        root=Path(directory)/'run';report=train(root);rows=list(csv.DictReader((root/'metrics.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');epochs=[int(r['epoch']) for r in rows]
    for key,label in [('train_loss','Train eval'),('val_loss','Validation')]:axes[0,0].semilogy(epochs,[float(r[key]) for r in rows],label=label)
    axes[0,0].set(title='Recall model: CE',xlabel='Epoch',ylabel='CE (log)');axes[0,0].legend(fontsize=9)
    axes[0,1].plot(epochs,[float(r['val_accuracy']) for r in rows],label='Value accuracy',color=BLUE)
    axes[0,1].plot(epochs,[float(r['val_key_match']) for r in rows],label='Attention key-match',color=ORANGE,ls='--')
    axes[0,1].axhline(1,color=GREEN,ls=':',label='Dict rule 100%')
    axes[0,1].set(title='Validation: hai metric khác nhau',xlabel='Epoch',ylim=(0,1.05));axes[0,1].legend(fontsize=9)
    probe=report['probe'];bars=axes[1,0].bar([f'key{k}' for k in probe['keys']],probe['attention'],color=[ORANGE if k==probe['query'] else BLUE for k in probe['keys']])
    axes[1,0].bar_label(bars,fmt='%.3f',padding=3);axes[1,0].set(title=f"Probe query{probe['query']}: attention sau học",ylabel='Weight',ylim=(0,1.15))
    heat(axes[1,1],report['test']['confusion'],'Test sau chọn best checkpoint',['value0','value1','value2'],['pred0','pred1','pred2'],fmt='.0f')
    fig.suptitle(f"Single-head cross-attention 235 parameter; best epoch {report['best_epoch']}; synthetic test64 sample /32 group")
    save(fig,'07-training');print('Created 7 PNG + 7 SVG')

if __name__=='__main__':main()
