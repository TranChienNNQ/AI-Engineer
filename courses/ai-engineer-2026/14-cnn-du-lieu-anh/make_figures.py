"""Five original figures from hand arrays and locally generated image/training artifacts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from lab import train, read_gray, hand_demo
OUT=Path(__file__).resolve().parent/'figures'

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150);fig.savefig(OUT/(name+'.svg'));plt.close(fig)

def matrix(ax,array,title):
    array=np.asarray(array);ax.imshow(array,cmap='Blues');ax.set_title(title)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):ax.text(j,i,f'{array[i,j]:g}',ha='center',va='center',color='white' if array[i,j]>(array.min()+array.max())/2 else 'black')
    ax.set_xticks(range(array.shape[1]));ax.set_yticks(range(array.shape[0]))

def main():
    OUT.mkdir(exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    with TemporaryDirectory(prefix='lesson14-figures-') as directory:
        root=Path(directory)/'run';report=train(root)
        import json
        records=json.loads((root/'data'/'manifest.json').read_text())['records']
        fig,axes=plt.subplots(2,4,figsize=(10,5.2),layout='constrained')
        for label in (0,1):
            selected=[r for r in records if r['split']=='train' and r['label']==label][:4]
            for ax,row in zip(axes[label],selected):
                ax.imshow(read_gray(root/'data'/row['path'])[0],cmap='gray',vmin=0,vmax=1)
                ax.set_title(f"{'Dọc' if label==0 else 'Ngang'}: group {row['group']}");ax.axis('off')
        fig.suptitle('PNG grayscale 16×16 → tensor (1,16,16) → batch (N,1,16,16)')
        save(fig,'01-images')
        example=read_gray(root/'data'/next(r['path'] for r in records if r['label']==0))
        rows=list(csv.DictReader((root/'metrics.csv').open()))
    hand=hand_demo()
    fig,axes=plt.subplots(1,4,figsize=(12,3.6),layout='constrained')
    matrix(axes[0],[[1,2,0],[0,1,3],[2,1,0]],'Ảnh X')
    axes[0].add_patch(Rectangle((-.5,-.5),2,2,fill=False,edgecolor='#ea580c',lw=3))
    matrix(axes[1],[[1,0],[0,-1]],'Kernel K; bias=0.5')
    matrix(axes[2],hand['convolution'],'Z: cross-correlation + bias')
    matrix(axes[3],hand['relu'],'ReLU; max pool2 → 1.5')
    save(fig,'02-convolution')
    fig,ax=plt.subplots(figsize=(11,4.7),layout='constrained');ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    nodes=[(.1,'Input\n1×16×16\nr=1, j=1'),(.36,'Conv3 + ReLU\n4×16×16\nPool2 → 4×8×8\nr=4, j=2'),
           (.65,'Conv3 + ReLU\n8×8×8\nPool2 → 8×4×4\nr=10, j=4'),(.91,'Flatten128\nLinear → 2\nlogits')]
    for x,text in nodes:ax.text(x,.57,text,ha='center',va='center',linespacing=1.6,bbox={'boxstyle':'round,pad=.6','facecolor':'#dbeafe'})
    for left,right in [(.19,.25),(.47,.54),(.76,.84)]:ax.annotate('',xy=(right,.57),xytext=(left,.57),arrowprops={'arrowstyle':'->','lw':2})
    ax.text(.5,.12,'Parameter: Conv1 40 + Conv2 296 + Linear 258 = 594\nr: receptive field một ô; j: khoảng cách trên ảnh gốc',ha='center',linespacing=1.7)
    ax.set_title('Một sample: giữ trục không gian đến sau hai block Conv/Pool')
    save(fig,'03-architecture')
    fig=plt.figure(figsize=(12,4.5),layout='constrained');grid=fig.add_gridspec(1,4,width_ratios=[1.7,1,1,1])
    ax=fig.add_subplot(grid[0]);ax.axis('off')
    ax.text(.5,.5,'64 group × 2 view\n↓ split theo group\nTrain: 40 group / 80 ảnh\nVal: 12 group / 24 ảnh\nTest: 12 group / 24 ảnh\n\nStats chỉ fit train\nRandom flip chỉ ở train',ha='center',va='center',linespacing=1.7,bbox={'boxstyle':'round,pad=.6','facecolor':'#dcfce7'})
    for index,image,title in [(1,example,'Gốc: nhãn dọc'),(2,torch.flip(example,[-1]),'Flip ngang: vẫn dọc'),(3,torch.rot90(example,1,[-2,-1]),'Rotate90: thành ngang')]:
        ax=fig.add_subplot(grid[index]);ax.imshow(image[0],cmap='gray',vmin=0,vmax=1);ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle('Tránh leakage và kiểm ý nghĩa nhãn sau biến đổi');save(fig,'04-data-pipeline')
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    epochs=[int(r['epoch']) for r in rows]
    for key,label in [('train_loss','Train eval'),('val_loss','Validation')]:axes[0].semilogy(epochs,[float(r[key]) for r in rows],label=label)
    axes[0].set(title='CE của model cố định',xlabel='Epoch',ylabel='Cross-entropy (log)');axes[0].legend(fontsize=9)
    for key,label in [('train_accuracy','Train eval'),('val_accuracy','Validation')]:axes[1].plot(epochs,[float(r[key]) for r in rows],label=label)
    axes[1].set(title='Accuracy và CE khác nhau',xlabel='Epoch',ylabel='Accuracy',ylim=(0,1.05));axes[1].legend(fontsize=9)
    matrix(axes[2],report['test']['confusion'],'Test: sau chọn checkpoint')
    axes[2].set_xticks([0,1],['Dọc','Ngang']);axes[2].set_yticks([0,1],['Dọc','Ngang']);axes[2].set(xlabel='Predicted',ylabel='True')
    fig.suptitle(f"CNN 594 parameter; best epoch {report['best_epoch']}; synthetic test 24 ảnh / 12 group")
    save(fig,'05-training');print('Created 5 PNG + 5 SVG')

if __name__=='__main__':main()
