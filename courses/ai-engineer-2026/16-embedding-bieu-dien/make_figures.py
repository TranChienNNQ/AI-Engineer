"""Five original embedding figures, including full-dimensional cosine from the actual CPU lab."""
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lab import train,load_best,setup,TicketClassifier,TOPICS
OUT=Path(__file__).resolve().parent/'figures'
BLUE='#2563eb';ORANGE='#ea580c'

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150);fig.savefig(OUT/(name+'.svg'));plt.close(fig)

def matrix(ax,array,title,rowlabels=None):
    a=np.asarray(array);ax.imshow(a,cmap='Blues');ax.set_title(title)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j,i,f'{a[i,j]:g}',ha='center',va='center',color='white' if a[i,j]>(a.min()+a.max())/2 else 'black')
    ax.set_xticks(range(a.shape[1]));ax.set_yticks(range(a.shape[0]))
    if rowlabels is not None:ax.set_yticklabels(rowlabels)


def main():
    OUT.mkdir(exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    ids=torch.tensor([2,3,2,0]);E=torch.tensor([[0.,0.],[0.,0.],[1.,0.],[0.,2.],[1.,1.]])
    fig,axes=plt.subplots(1,3,figsize=(11,4.2),layout='constrained')
    matrix(axes[0],torch.nn.functional.one_hot(ids,5).numpy(),'One-hot IDs [2,3,2,0]')
    matrix(axes[1],E.numpy(),'E: 5 hàng × 2 chiều',['PAD0','UNK1','login2','invoice3','network4'])
    matrix(axes[2],E[ids].numpy(),'Lookup = one-hot @ E')
    fig.suptitle('Bảng toy đặt tay; ID là chỉ số hàng, không phải số đo');save(fig,'01-lookup')
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    axes[0].annotate('',xy=(10,10),xytext=(0,0),arrowprops={'arrowstyle':'->','lw':2,'color':ORANGE})
    axes[0].annotate('',xy=(1,0),xytext=(0,0),arrowprops={'arrowstyle':'->','lw':3,'color':BLUE})
    axes[0].text(9.7,10.3,'b=(10,10)',ha='right',color=ORANGE)
    axes[0].text(1.1,.3,'q=a=(1,0)',color=BLUE)
    axes[0].set(xlim=(-.5,11),ylim=(-1,11),xlabel='Chiều 1',ylabel='Chiều 2',title='Dot bị ảnh hưởng bởi norm',aspect='equal');axes[0].grid(alpha=.2)
    axes[1].axis('off')
    table=axes[1].table(cellText=[['1','10'],['1','0.707107'],['0','13.453624']],colLabels=['a','b'],rowLabels=['Dot với q','Cosine với q','L2 distance'],loc='center',cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(11);table.scale(.8,2)
    axes[1].set_title('Cùng vector, ranking khác theo phép đo')
    axes[1].text(.5,.15,'Unit vectors: d² = 2 − 2cos\nZero vector không có hướng cosine',ha='center',transform=axes[1].transAxes,linespacing=1.5)
    save(fig,'02-geometry')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].axis('off');axes[0].text(.5,.55,'[1,0] + [0,2] + [1,0]\n↓ chia 3 token thật\np = [2/3, 2/3]\n\nChia cả PAD: [1/2,1/2]\nL = ½ ||p||² = 4/9',ha='center',va='center',linespacing=1.7,bbox={'boxstyle':'round,pad=.7','facecolor':'#dbeafe'})
    axes[0].set_title('PAD có vector0 vẫn cần mask mẫu số')
    bars=axes[1].bar(['PAD0','UNK1','login2','invoice3','network4'],[0,0,4/9,2/9,0],color=['gray','gray',BLUE,ORANGE,'gray'])
    axes[1].bar_label(bars,fmt='%.4f',padding=3);axes[1].set(ylim=(0,.53),ylabel='Gradient mỗi tọa độ',title='Hàng lặp: cộng hai contribution')
    axes[1].tick_params(axis='x',labelsize=9);save(fig,'03-pooling-gradient')
    with TemporaryDirectory(prefix='lesson16-figures-') as directory:
        root=Path(directory)/'run';report=train(root);model,checkpoint=load_best(root/'best.pt')
        rows=list(csv.DictReader((root/'metrics.csv').open()))
        setup();initial=TicketClassifier(len(checkpoint['vocab']),checkpoint['config']['dim'])
        tokens=[t for group in TOPICS for t in group]+['issue']
        indices=[checkpoint['vocab'][t] for t in tokens]
        arrays=[]
        for current in [initial,model]:
            with torch.no_grad():
                unit=torch.nn.functional.normalize(current.embedding.weight[indices],dim=1)
                arrays.append((unit@unit.T).numpy())
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),layout='constrained')
    for ax,a,title in zip(axes,arrays,['Khởi tạo','Sau học classification']):
        im=ax.imshow(a,cmap='coolwarm',vmin=-1,vmax=1)
        ax.set_xticks(range(len(tokens)),tokens,rotation=90);ax.set_yticks(range(len(tokens)),tokens)
        for boundary in [2.5,5.5,8.5]:ax.axhline(boundary,color='black',lw=.7);ax.axvline(boundary,color='black',lw=.7)
        ax.set_title(title);ax.tick_params(labelsize=9)
    fig.colorbar(im,ax=axes,shrink=.8,label='Cosine trong đủ 4 chiều')
    fig.suptitle('Học theo topic không bảo đảm mọi hàng gần nhau là đồng nghĩa');save(fig,'04-learned-space')
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained');epochs=[int(r['epoch']) for r in rows]
    for key,label in [('train_loss','Train eval'),('val_loss','Validation')]:axes[0].semilogy(epochs,[float(r[key]) for r in rows],label=label)
    axes[0].set(title='CE sau mỗi epoch',xlabel='Epoch',ylabel='CE (log)');axes[0].legend(fontsize=9)
    axes[1].plot(epochs,[float(r['val_accuracy']) for r in rows],label='Model validation',color=BLUE)
    axes[1].axhline(1/3,color='gray',ls='--',label='Majority ≈ 33.33%')
    axes[1].axhline(1,color='#15803d',ls=':',label='Keyword rule = 100%')
    axes[1].set(title='So với baseline định trước',xlabel='Epoch',ylabel='Accuracy',ylim=(0,1.05));axes[1].legend(fontsize=8)
    matrix(axes[2],report['test']['confusion'],'Test sau chọn checkpoint',['Auth','Billing','Network'])
    axes[2].set_xticks([0,1,2],['Auth','Billing','Network'],rotation=30);axes[2].set(xlabel='Predicted',ylabel='True')
    fig.suptitle(f"Embedding+mean+head: 99 parameter; best epoch {report['best_epoch']}; synthetic test54 ticket")
    save(fig,'05-training');print('Created 5 PNG + 5 SVG')

if __name__=='__main__':main()
