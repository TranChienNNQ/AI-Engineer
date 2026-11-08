"""Seven original figures; measured chart uses an actual lab report, others equations/simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path);p.add_argument('--out',type=Path,default=Path(__file__).parent/'figures');a=p.parse_args()
    from recorded_measurements import figure_data
    r=json.loads(a.report.read_text()) if a.report else figure_data();a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    blue,green,orange,red='#245ca6','#25836a','#dd8a21','#be4b53'
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(a.out/f'{name}.{ext}',dpi=155,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    def box(ax,x,y,text,w=.22,h=.2,color=blue):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=color+'14',ec=color,lw=1.4))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
    def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':1.7,'color':'#566577'})
    fig,ax=plt.subplots(figsize=(12,5));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    ax.set_title('P=4, N=3: prefill đã tạo token đầu tiên',fontsize=17,pad=20)
    box(ax,.03,.50,'Prefill\ninput: x₀ x₁ x₂ x₃\ncache length: 4',w=.24,h=.29,color=blue)
    box(ax,.39,.50,'Decode 1\ninput: y₁\ncache length: 5',w=.24,h=.29,color=green)
    box(ax,.75,.50,'Decode 2\ninput: y₂\ncache length: 6',w=.22,h=.29,color=green)
    arrow(ax,.27,.64,.39,.64);arrow(ax,.63,.64,.75,.64)
    for x,t in [(.15,'phát y₁'),(.51,'phát y₂'),(.86,'phát y₃ → dừng')]:
        arrow(ax,x,.50,x,.34);ax.text(x,.27,t,ha='center',color=blue,fontsize=13)
    ax.text(.5,.08,'Không cần forward y₃ nếu request kết thúc. Cache giữ P + N − 1 = 6 vị trí.',ha='center',fontsize=12)
    save(fig,'01-prefill-decode')

    matrix=np.array([[1,1,1,1,0],[1,1,1,1,1],[1,1,0,0,0],[1,1,1,0,0]])
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');ax.imshow(matrix,cmap=matplotlib.colors.ListedColormap(['#e8edf3','#b6d9cc']),vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(5),[f'key{k}' for k in range(5)]);ax.set_yticks(range(4),['A query pos3','A query pos4','B query pos1','B query pos2'])
    for i in range(4):
        for j in range(5):ax.text(j,i,'Cho phép' if matrix[i,j] else 'Chặn',ha='center',va='center',fontsize=11)
    ax.set(xlabel='Vị trí key tuyệt đối: không lấy padded width làm position',ylabel='A: past3, new2\nB: past1, new2')
    ax.set_title('Offset causal + độ dài riêng từng request',fontsize=16,pad=15)
    save(fig,'02-offset-mask')

    t=np.arange(256,8193,256);fig,axs=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained')
    for h,c,label in [(32,blue,'MHA: Hkv=32'),(8,green,'GQA: Hkv=8')]:
        mem=2*32*h*128*t*2/1024**3;axs[0].plot(t,mem,color=c,label=label,lw=2)
    axs[0].set(xlabel='Context token / request',ylabel='Payload KV, GiB');axs[0].legend();axs[0].set_title('Giả định: L32, dh128, 2 byte, B1',fontsize=11)
    axs[1].bar(['Đã dùng','Cấp sẵn'],[2,8],color=[green,orange]);axs[1].set(ylabel='KiB (chỉ buffer K/V)',ylim=(0,9))
    axs[1].set_title('Lab: lengths[3,5], capacity16/hàng',fontsize=11)
    for i,v in enumerate([2,8]):axs[1].text(i,v+.2,str(v),ha='center')
    fig.suptitle('KV memory: persistent payload khác tổng bộ nhớ process',fontsize=15)
    save(fig,'03-kv-memory')

    fig,axs=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    for ax,lengths,caps,labels,title in [(axs[0],[2,9,3,8],[9,9,8,8],['r0','r1','r2','r3'],'Theo thứ tự đến: 34 ô'),(axs[1],[2,3,8,9],[3,3,9,9],['r0','r2','r3','r1'],'Bucket độ dài: 24 ô')]:
        for row,(n,cap) in enumerate(zip(lengths,caps)):
            for col in range(cap):ax.barh(row,1,left=col,color=blue if col<n else '#dde3eb',edgecolor='white',height=.72)
        ax.set_yticks(range(4),[f'{name}: {n}' for name,n in zip(labels,lengths)]);ax.invert_yaxis();ax.set_xticks(range(10));ax.set(xlim=(0,9.5),xlabel='Ô token; xanh = thật, xám = PAD');ax.set_title(title,fontsize=13)
    fig.suptitle('B=2: cùng 22 token thật, padding giảm 12 → 2',fontsize=16)
    save(fig,'04-padding-bucketing')

    fig,axs=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained');colors=dict(A=blue,B=orange,C=green,D=red)
    for ax,key,title in [(axs[0],'static','Static cohort'),(axs[1],'continuous','Continuous')]:
        data=r['schedule'][key]
        for row,job in enumerate(r['schedule']['jobs']):
            for it in data['timeline']:
                if job['name'] in it['active']:ax.barh(row,1,left=it['tick'],color=colors[job['name']],edgecolor='white',height=.6)
            ax.plot(job['arrival'],row,'>',color='black',ms=7)
        ax.set_yticks(range(4),['A','B','C','D']);ax.invert_yaxis();ax.set_xticks(range(7));ax.set(xlim=(-.15,6.3),xlabel='Tick giả định; ▶ là arrival');ax.set_title(f"{title}: xong tick{data['makespan']}",fontsize=13)
    fig.suptitle('2 slot, 1 token/request/tick; bỏ qua prefill và hardware cost',fontsize=14)
    save(fig,'05-scheduling')

    rows=r['benchmark']['rows'];xx=np.arange(4);med=np.array([v['median_ms'] for v in rows]);lo=np.array([v['min_ms'] for v in rows]);hi=np.array([v['max_ms'] for v in rows])
    labels=['Single\nfull','Single\ncache','Batch\ncache','Batch\ncache SDPA'];fig,axs=plt.subplots(1,2,figsize=(11.5,5),layout='constrained')
    axs[0].bar(xx,med,yerr=np.stack([med-lo,hi-med]),color=[blue,green,orange,red],capsize=4)
    axs[0].set_xticks(xx,labels);axs[0].set_ylabel('Makespan 4 request, ms');axs[0].set_ylim(0,max(hi)*1.18)
    for i,v in enumerate(med):axs[0].text(i,hi[i]+.45,f'{v:.2f}',ha='center',fontsize=10)
    axs[1].bar(xx,[v['output_tokens_per_second'] for v in rows],color=[blue,green,orange,red]);axs[1].set_xticks(xx,labels);axs[1].set_ylabel('Aggregate output token/s')
    fig.suptitle('Một lượt CPU: 4 × (48 prompt + 16 output), median7repeat; range min–max',fontsize=13)
    save(fig,'06-cpu-benchmark')

    fig,ax=plt.subplots(figsize=(12,5.3));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title('Tối ưu inference là một vòng đo có kiểm chứng',fontsize=17,pad=15)
    box(ax,.03,.65,'Khóa workload\nmodel, tokens, budget',w=.25,color=blue)
    box(ax,.38,.65,'Kiểm đúng\nlogits, mask, lifecycle',w=.25,color=green)
    box(ax,.73,.65,'Đo bottleneck\nqueue, compute, memory',w=.24,color=orange)
    arrow(ax,.28,.75,.38,.75);arrow(ax,.63,.75,.73,.75)
    box(ax,.73,.22,'Thử một thay đổi\ncache / batch / backend',w=.24,color=red)
    box(ax,.38,.22,'Đo lại cùng điều kiện\nTTFT, ITL, goodput, KV',w=.25,color=blue)
    box(ax,.03,.22,'Giữ hoặc rollback\nđúng + đạt yêu cầu',w=.25,color=green)
    arrow(ax,.85,.65,.85,.42);arrow(ax,.73,.32,.63,.32);arrow(ax,.38,.32,.28,.32)
    ax.text(.5,.04,'Số vị trí ít hơn chưa là latency thấp hơn; output giống chưa là benchmark chất lượng.',ha='center',fontsize=11)
    save(fig,'07-optimization-loop')
    print(json.dumps({'figures':7,'formats':['png','svg'],'out':str(a.out)},ensure_ascii=False))


if __name__=='__main__':main()
