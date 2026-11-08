"""Reproduce static learning figures. Timing figure is tied to the supplied report."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE='#255eaa';GREEN='#20866e';ORANGE='#de8a1d';RED='#bb4955';GRAY='#57677c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})


def box(ax,x,y,w,h,label,color=BLUE,size=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',edgecolor=color,facecolor=color+'12',linewidth=1.8))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)


def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.8))


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).with_name('figures'));a=p.parse_args()
    r=json.loads(a.report.read_text());a.out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(a.out/(name+'.png'),dpi=145,bbox_inches='tight',facecolor='white')
        fig.savefig(a.out/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    fig.suptitle('Một công thức đi qua nhiều lớp trước khi phần cứng thực thi',fontsize=17)
    labels=[('Request / model graph','Latency toàn request, batch và dữ liệu'),('Operator: matmul → bias → ReLU','Shape, dtype, stride và semantics'),('Kernel / library implementation','Fusion, tile, layout và buffers'),('Instructions / hardware','SIMD, matrix instructions và memory hierarchy')]
    for i,(label,note) in enumerate(labels):
        y=.79-i*.22;box(ax,.08,y,.45,.13,label,[BLUE,GREEN,ORANGE,RED][i]);ax.text(.59,y+.065,note,va='center',fontsize=11)
        if i<3:arrow(ax,(.305,y-.005),(.305,y-.08))
    save(fig,'01-compute-stack')

    fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.35,1]});ax=axes[0];ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    fig.suptitle('Fusion giữ intermediate cục bộ thay vì ghi thành hai mảng',fontsize=16)
    for i,label in enumerate(['x → t = a·x','t → u = t+b','u → y = ReLU(u)']):
        y=.74-i*.18;box(ax,.12,y,.76,.12,label,BLUE)
        if i<2:arrow(ax,(.5,y),(.5,y-.05))
    box(ax,.12,.08,.76,.14,'Fused: x[i] → local z → y[i]',GREEN)
    ax=axes[1];ax.bar(['Three pass','Fused'],[24,8],color=[BLUE,GREEN],width=.55)
    ax.set(ylabel='Logical array traffic, MB',ylim=(0,29),title='N = 1.000.000 FP32; a,b scalar')
    for i,v in enumerate([24,8]):ax.text(i,v+.6,str(v),ha='center',fontsize=14)
    fig.text(.5,-.01,'Source-level ledger; chưa là measured DRAM/HBM bytes hoặc speedup.',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.04,1,.94));save(fig,'02-fusion-traffic')

    fig,axes=plt.subplots(1,3,figsize=(12,4));fig.suptitle('Tile theo trục K: cộng hai kết quả partial, có tail K=1',fontsize=16)
    matrices=[np.array([[1,2,3],[4,5,6]]),np.array([[1,0],[0,1],[1,1]]),np.array([[4,5],[10,11]])]
    for ax,mat,title in zip(axes,matrices,['A [2,3]','B [3,2]','C [2,2]']):
        ax.imshow(np.zeros_like(mat),cmap='Blues',vmin=-1,vmax=3)
        for (i,j),v in np.ndenumerate(mat):ax.text(j,i,str(v),ha='center',va='center',fontsize=17)
        ax.set(xticks=np.arange(mat.shape[1]),yticks=np.arange(mat.shape[0]),title=title)
        if title.startswith('A'):ax.axvline(1.5,color=ORANGE,lw=3)
        if title.startswith('B'):ax.axhline(1.5,color=ORANGE,lw=3)
    fig.text(.5,.02,'K đầu: [[1,2],[4,5]]   +   K cuối: [[3,3],[6,6]]   =   [[4,5],[10,11]]',ha='center',fontsize=12)
    fig.tight_layout(rect=(0,.09,1,.91));save(fig,'03-matmul-tiles')

    fig,ax=plt.subplots(figsize=(10,5));x=np.logspace(-2,3,300);y=np.minimum(100,1*x)
    ax.loglog(x,y,color=BLUE,lw=2.5,label='min(100 TFLOP/s, 1 TB/s × intensity)')
    ax.axvline(100,ls='--',color=GRAY,alpha=.6);ax.scatter([.25],[.25],color=GREEN,s=65,zorder=4)
    ax.annotate('Affine fused: I=0,25\ntrần 0,25 TFLOP/s',xy=(.25,.25),xytext=(.8,.055),arrowprops=dict(arrowstyle='->'),fontsize=12)
    ax.text(8,45,'Memory ceiling',color=BLUE);ax.text(140,120,'Compute ceiling',color=BLUE)
    ax.set(xlabel='Arithmetic intensity, FLOP/byte',ylabel='Attainable throughput ceiling, TFLOP/s',title='Roofline GIẢ ĐỊNH: P=100 TFLOP/s, B=1 TB/s; không số đo máy')
    ax.grid(True,which='both',alpha=.15);ax.legend(loc='upper left',fontsize=10);fig.tight_layout();save(fig,'04-roofline')

    fig,axes=plt.subplots(1,2,figsize=(12,4.8));fig.suptitle('Online attention: rescale state khi m tăng, rồi mới cộng khối mới',fontsize=15)
    ax=axes[0];ax.axis('off')
    table=ax.table(cellText=[['0','1','10','10'],['ln2','3/2','25','50/3'],['ln3','2','170/3','85/3']],
                   rowLabels=['key1','key2','key3'],colLabels=['m','l','z','z/l'],loc='center',cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(13);table.scale(1,2.3)
    ax.set_title('Scores [0,ln2,ln3]; values [10,20,40]',pad=10)
    ax=axes[1];ax.plot([1,2,3],[10,50/3,85/3],marker='o',color=GREEN,lw=2)
    ax.axhline(85/3,ls='--',color=GRAY,label='Output cuối = 85/3')
    ax.set(xticks=[1,2,3],xlabel='Số key đã đọc',ylabel='z/l của phần đã đọc',ylim=(0,34))
    ax.legend();fig.text(.5,.02,'α = exp(m cũ − m mới) phải nhân cả l cũ và z cũ.',ha='center',fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.92));save(fig,'05-online-attention')

    rows=r['benchmark']['rows'];fig,ax=plt.subplots(figsize=(11,5.3));x=np.arange(len(rows));width=.34
    for pos,mode,color in [(-.5,'separate',BLUE),(.5,'fused',GREEN)]:
        med=np.array([row['modes'][mode]['median_ns']/1000 for row in rows]);lo=np.array([row['modes'][mode]['p10_ns']/1000 for row in rows]);hi=np.array([row['modes'][mode]['p90_ns']/1000 for row in rows])
        bars=ax.bar(x+pos*width,med,width,label=mode+' C',color=color,yerr=np.vstack([med-lo,hi-med]),capsize=5)
        for bar,v in zip(bars,med):ax.text(bar.get_x()+bar.get_width()/2,v*1.22,f'{v:.3f}',ha='center',fontsize=10)
    ax.set_yscale('log');ax.set(xticks=x,xticklabels=[f"N={row['elements']:,}\nratio {row['median_ratio_separate_over_fused']:.2f}×" for row in rows],ylabel='µs/call, log scale',title='CPU warm: median và p10/p90 của 9 block averages')
    ax.set_ylim(top=max(row['modes']['separate']['p90_ns']/1000 for row in rows)*3);ax.legend(loc='upper left')
    fig.text(.5,.01,'Có Python/ctypes overhead; preallocated buffers; không GPU hoặc API latency benchmark.',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,'06-cpu-benchmark')

    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));fig.suptitle('Một tối ưu cần correctness và tác động tới workload mục tiêu',fontsize=16)
    items=[('1. Profile / baseline','Đo phần nào đang tốn thời gian',BLUE),('2. Giả thuyết','Fusion / tiling / layout / precision',GREEN),('3. Kiểm correctness','Shape, tail, mask, dtype, gradient',ORANGE),('4. Đo công bằng','Cold/warm, raw samples, boundaries',RED),('5. Đo lại end-to-end','Latency / goodput / quality / cost',BLUE)]
    for i,(label,note,color) in enumerate(items):
        y=.82-i*.18;box(ax,.06,y,.36,.12,label,color);ax.text(.48,y+.06,note,va='center',fontsize=12)
        if i<4:arrow(ax,(.24,y-.005),(.24,y-.055))
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'07-optimization-workflow')
    print(json.dumps(dict(png=7,svg=7,out=str(a.out))))


if __name__=='__main__':main()
