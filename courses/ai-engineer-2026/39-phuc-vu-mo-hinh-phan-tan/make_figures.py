"""Original diagrams + hypothetical models + actual local-process report outcomes."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).parent/'figures');a=p.parse_args()
    r=json.loads(a.report.read_text());a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    blue,green,orange,red='#245ca6','#25836a','#dd8a21','#be4b53'
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(a.out/f'{name}.{ext}',dpi=155,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    def box(ax,x,y,text,w=.24,h=.18,color=blue):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=color+'14',ec=color,lw=1.5))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
    def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':1.7,'color':'#566577'})
    fig,ax=plt.subplots(figsize=(12,6.4));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title('Nhân bản model và chia model giải hai bài toán khác nhau',fontsize=16,pad=15)
    ax.text(.02,.84,'Replica',fontsize=13,weight='bold',color=blue)
    box(ax,.22,.73,'Request A\nFull model R0',color=blue);box(ax,.65,.73,'Request B\nFull model R1',color=blue)
    ax.text(.02,.51,'Tensor\nparallel',fontsize=13,weight='bold',color=green)
    box(ax,.22,.42,'Cùng request\nShard0: partial A',color=green);box(ax,.65,.42,'Cùng request\nShard1: partial B',color=green)
    arrow(ax,.46,.51,.54,.51);ax.text(.56,.51,'Σ',ha='center',va='center',fontsize=18,color=green);arrow(ax,.65,.51,.59,.51)
    ax.text(.02,.20,'Pipeline\nparallel',fontsize=13,weight='bold',color=orange)
    box(ax,.22,.11,'Request X\nStage0: layers đầu',color=orange);box(ax,.65,.11,'Cùng request X\nStage1: layers sau',color=orange)
    arrow(ax,.46,.20,.65,.20);ax.text(.55,.25,'activation',ha='center',fontsize=10)
    save(fig,'01-parallelism-map')

    fig,ax=plt.subplots(figsize=(12,5.2));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title('TP tính tay: H=[1,1,0,3], bias output cộng một lần',fontsize=16,pad=18)
    box(ax,.04,.60,'Rank0: hidden[1,1]\npartial [0,3]\nweights 56 byte',w=.30,h=.25,color=green)
    box(ax,.66,.60,'Rank1: hidden[0,3]\npartial [3,−6]\nweights 56 byte',w=.30,h=.25,color=blue)
    box(ax,.33,.16,'Coordinator: sum + [.5,−.5]\nY=[3.5,−3.5]\nbias 8 byte',w=.34,h=.25,color=orange)
    arrow(ax,.19,.60,.40,.41);arrow(ax,.81,.60,.60,.41)
    ax.text(.5,.04,'Reduce về coordinator trong lab; chưa all-reduce hoặc NCCL.',ha='center',fontsize=12)
    save(fig,'02-tensor-hand')

    fig,axs=plt.subplots(1,2,figsize=(12.5,5),layout='constrained');colors=[blue,green,orange,red]
    for ax,key,times,title in [(axs[0],'pipeline_balanced',[2,2,2],'Balanced [2,2,2]ms'),(axs[1],'pipeline_uneven',[2,4,2],'Uneven [2,4,2]ms')]:
        data=r['hand'][key]
        for mb,row in enumerate(data['finish']):
            for stage,end in enumerate(row):
                ax.barh(stage,times[stage],left=end-times[stage],height=.65,color=colors[mb],edgecolor='white')
                ax.text(end-times[stage]/2,stage,f'm{mb}',ha='center',va='center',color='white',fontsize=9)
        ax.set_yticks(range(3),['Stage0','Stage1','Stage2']);ax.invert_yaxis();ax.set(xlabel='ms giả định, forward-only',xlim=(0,21))
        ax.set_title(f"{title}\nxong {data['makespan']:g}ms; slot use {100*data['stage_slot_utilization']:.1f}%",fontsize=12)
    fig.suptitle('4 microbatch: bubble và stage chậm làm giảm hiệu quả pipeline',fontsize=15)
    save(fig,'03-pipeline-timeline')

    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');m=r['hand']['memory']
    axs[0].bar([0,1],[7,m['sharded_per_rank_gib']],color=blue,label='Phần chia đều')
    axs[0].bar([1],[m['replicated_gib']],bottom=[m['sharded_per_rank_gib']],color=orange,label='Phần replicated')
    axs[0].set_xticks([0,1],['28/4 ngây thơ','90%shard + 10%replica']);axs[0].set(ylabel='Weight GiB / rank',ylim=(0,12));axs[0].legend(fontsize=9)
    axs[0].text(0,7.25,'7',ha='center');axs[0].text(1,9.35,'9.1',ha='center');axs[0].set_title('Ledger TP4 giả định; chưa KV/workspace',fontsize=11)
    bw=np.array([1,2,4,8]);time_ms=(6*10e-6+1.5*16*1024**2/(bw*1024**3))*1000
    axs[1].plot(bw,time_ms,'o-',color=green);axs[1].set(xlabel='Băng thông hiệu dụng giả định, GiB/s',ylabel='Ring all-reduce model, ms');axs[1].set_xticks(bw)
    axs[1].annotate('2.990ms tại 8GiB/s',(8,time_ms[-1]),xytext=(3.5,11),arrowprops={'arrowstyle':'->'},fontsize=10)
    axs[1].set_title('p4, payload16MiB, α10µs; không benchmark',fontsize=11)
    fig.suptitle('Fit memory và communication cost phải được kiểm riêng',fontsize=15)
    save(fig,'04-memory-communication')

    fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
    ax.barh(1,1,left=0,height=.45,color=blue);ax.barh(1,3,left=1,height=.45,color=red)
    ax.barh(0,3,left=0,height=.45,color=orange);ax.barh(0,1,left=3,height=.45,color=green)
    ax.text(.5,1,'RUNNING',ha='center',va='center',color='white',fontsize=10);ax.text(2.5,1,'TIMEOUT giữ nguyên',ha='center',va='center',color='white',fontsize=11)
    ax.text(1.5,0,'BUSY: caller bỏ chờ chưa là compute đã dừng',ha='center',va='center',fontsize=10);ax.text(3.5,0,'FREE',ha='center',va='center',color='white')
    for x,label in [(0,'Submit/hold'),(1,'Deadline'),(2,'Release barrier'),(3,'Reply bị bỏ')]:ax.axvline(x,color='gray',ls=':',lw=1)
    ax.set_yticks([1,0],['Client record','Worker slot']);ax.set_xticks([0,1,2,3],['Submit/hold','Deadline','Release','Reply bị bỏ']);ax.set(xlim=(-.1,4.1),ylim=(-.7,1.7),xlabel='Thứ tự sự kiện trong fixture; không là tỷ lệ thời gian benchmark')
    ax.set_title('Timeout/cancel và thu hồi slot là hai trạng thái cần tách',fontsize=16,pad=18)
    save(fig,'05-deadline-slot')

    counts=Counter(v['status'] for v in r['router']['snapshot']['records']);fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    axs[0].axis('off');data=[[f'x{i}',*map(str,out)] for i,out in enumerate(r['shards']['reference'])]
    table=axs[0].table(cellText=data,colLabels=['Input','y₀','y₁'],cellLoc='center',loc='center');table.auto_set_font_size(False);table.set_fontsize(12);table.scale(1,2)
    axs[0].set_title('Replica / TP / PP cùng outputs\nMax error0 trên fixture nhỏ',fontsize=13)
    labels=['OK','TIMEOUT','CANCELLED','WORKER_LOST','BAD_REPLY'];values=[counts[k] for k in labels];xx=np.arange(5)
    axs[1].bar(xx,values,color=[green,orange,orange,red,red]);axs[1].set_xticks(xx,['OK','TIMEOUT','CANCEL','WORKER\nLOST','BAD\nREPLY'],fontsize=10);axs[1].set(ylabel='Terminal records / 10',ylim=(0,7))
    for i,v in enumerate(values):axs[1].text(i,v+.15,str(v),ha='center')
    axs[1].set_title('Logical outcomes trong coordinator\n2 late replies bị bỏ; 1 stale frame tổng hợp',fontsize=12)
    fig.suptitle('Kết quả local-process lab; chưa đo GPU speed hoặc network failure',fontsize=14)
    save(fig,'06-lab-outcomes')

    fig,ax=plt.subplots(figsize=(12.5,6));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title('Chọn hướng thử theo ràng buộc rồi kiểm cùng workload/SLO',fontsize=16,pad=18)
    cases=[(.68,'Model không fit','Footprint / TP / PP','Memory/rank, layout\ncommunication',blue),(.38,'Thiếu throughput','Replica + batch/routing','TTFT / ITL / goodput\nKV và admission',green),(.08,'Lỗi / cập nhật model','Group identity\nreadiness / drain','Fault injection\ncleanup / rollback',orange)]
    for y,left,middle,right,c in cases:
        box(ax,.03,y,left,w=.25,h=.19,color=c);box(ax,.38,y,middle,w=.25,h=.19,color=c);box(ax,.73,y,right,w=.24,h=.19,color=c)
        arrow(ax,.28,y+.095,.38,y+.095);arrow(ax,.63,y+.095,.73,y+.095)
    save(fig,'07-decision-map')
    print(json.dumps({'figures':7,'formats':['png','svg'],'out':str(a.out)},ensure_ascii=False))


if __name__=='__main__':main()
