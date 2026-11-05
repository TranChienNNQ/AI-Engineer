"""Rebuild five figures from the executed CPU lab."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from lab import train_toy
OUT=Path(__file__).resolve().parent/'figures'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150);fig.savefig(OUT/(name+'.svg'));plt.close(fig)
def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for ax,a,title in zip(axes,[np.array([[1,2],[3,4]]),np.array([[1,4],[9,16]]),np.array([[7,10],[15,22]])],['X: shape (2,2)','X*X: theo phần tử','X@X: nhân ma trận']):
        ax.imshow(a,cmap='Blues');ax.set_title(title)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(a[i,j]),ha='center',va='center',color='white' if a[i,j]>a.max()/2 else 'black')
        ax.set_xticks([0,1]);ax.set_yticks([0,1])
    fig.suptitle('Tensor: shape, dtype=float64, device=CPU');save(fig,'01-tensor')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.axis('off')
    nodes=[(.12,.5,'w=0.5, b=1\nx=2'),(.4,.5,'a=wx+b=2'),(.7,.8,'a²=4\ngóp 4'),(.7,.2,'3a=6\ngóp 3'),(.92,.5,'L=10')]
    for a,b in [(0,1),(1,2),(1,3),(2,4),(3,4)]:
        ax.annotate('',xy=nodes[b][:2],xytext=nodes[a][:2],arrowprops={'arrowstyle':'->','color':'gray'})
    for x,y,t in nodes:ax.text(x,y,t,ha='center',va='center',bbox={'boxstyle':'round','facecolor':'#dbeafe'})
    ax.set_title('Autograd: da=4+3=7, dw=14, db=7');save(fig,'02-autograd')
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    x=np.arange(3);ax.bar(x-.18,[14,28,14],width=.35,label='w.grad');ax.bar(x+.18,[7,14,7],width=.35,label='b.grad')
    ax.set_xticks(x,['Backward 1','Graph mới, chưa reset','Reset rồi graph mới']);ax.set_ylabel('Gradient buffer');ax.legend();ax.set_title('Graph mới không tự xóa gradient leaf');save(fig,'03-gradient-buffer')
    fig,ax=plt.subplots(figsize=(10,3.8),layout='constrained');ax.axis('off')
    for i,t in enumerate(['Dataset\n5 sample','Batch 1\nID 0,1','Batch 2\nID 2,3','Batch 3\nID 4']):
        ax.text(.12+i*.25,.55,t,ha='center',bbox={'boxstyle':'round,pad=.8','facecolor':'#dcfce7' if i<3 else '#ffedd5'})
    ax.text(.5,.15,'drop_last=False: 2+2+1=5 mẫu; drop_last=True: chỉ 4 mẫu',ha='center');ax.set_title('DataLoader batch_size=2, shuffle=False');save(fig,'04-dataloader')
    model,history=train_toy()
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].semilogy(np.arange(len(history)),history);axes[0].set(title='Đo lại model cố định sau mỗi epoch',xlabel='Epoch',ylabel='Train MSE (log)')
    x=np.arange(5);axes[1].scatter(x,2*x+1,label='Train tự tạo');axes[1].plot(x,model.weight.item()*x+model.bias.item(),label='Linear đã fit');axes[1].legend();axes[1].set(xlabel='x',ylabel='y',title='5 mẫu train; chưa có validation/test')
    save(fig,'05-training')
    print('Created 5 PNG + 5 SVG')
if __name__=='__main__':main()
