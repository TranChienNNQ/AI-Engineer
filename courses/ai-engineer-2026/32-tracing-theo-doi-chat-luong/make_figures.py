"""Seven reproducible figures: actual SDK waterfall, labeled arithmetic and synthetic cohorts."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

ROOT=Path(__file__).resolve().parent
BLUE='#2863a5';GREEN='#227b62';RED='#bd4949';GRAY='#596777';GOLD='#d89a24'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})


def canvas(title,subtitle,size=(13,7.5)):
    fig,ax=plt.subplots(figsize=size);ax.set(xlim=(0,13),ylim=(0,7));ax.axis('off')
    fig.suptitle(title,x=.055,y=.96,ha='left',fontsize=18,fontweight='bold',color='#172d45')
    fig.text(.055,.89,subtitle,fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.04,right=.98,bottom=.08,top=.84)
    return fig,ax


def box(ax,x,y,w,h,label,color=BLUE,fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07,rounding_size=0.12',
                              facecolor=color+'12',edgecolor=color,linewidth=1.5))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color='#172d45',linespacing=1.6)


def arrow(ax,x,y,xx,yy,color=GRAY):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=15,color=color,lw=1.5))


def save(fig,name,out):
    for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=170,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'figures');args=p.parse_args()
    if args.report:r=json.loads(args.report.read_text())
    else:
        import asyncio
        from lab import report
        r=asyncio.run(report())
    out=args.out;out.mkdir(parents=True,exist_ok=True);q=r['quality'];h=r['hand']
    fig,ax=canvas('01 · Hai luồng dữ liệu, hai câu hỏi cần trả lời',
                  'Telemetry tìm đường lỗi/chậm; prediction + feedback giúp kiểm chất lượng có nhãn.')
    box(ax,.4,4.4,2.8,1.5,'API / QUEUE / MODEL\nOperations + version',BLUE,10)
    box(ax,4.7,4.4,3.1,1.5,'TRACE / LOG / METRIC\nPath / event / aggregate',BLUE,10)
    box(ax,9.2,4.4,3,1.5,'CHẨN ĐOÁN\nChậm ở đâu, lỗi tầng nào?',GREEN,10)
    arrow(ax,3.3,5.15,4.6,5.15);arrow(ax,7.9,5.15,9.1,5.15)
    box(ax,.4,1.5,2.8,1.5,'PREDICTION LEDGER\nID + score + version',GOLD,10)
    box(ax,4.7,1.5,3.1,1.5,'FEEDBACK JOIN\nLabel + available time',GOLD,10)
    box(ax,9.2,1.5,3,1.5,'QUALITY / SLICES\nĐúng không, nhóm nào kém?',GREEN,10)
    arrow(ax,3.3,2.25,4.6,2.25);arrow(ax,7.9,2.25,9.1,2.25)
    ax.text(6.5,.45,'Trace success không tự là prediction đúng. Quality ledger không phụ thuộc việc trace được sample.',ha='center',fontsize=11)
    save(fig,'01-signals-and-quality',out)

    spans=[s for s in r['tracing']['normal']['spans'] if s['trace_id']=='1'*32]
    order=['POST /v1/predict','queue.publish','queue.wait','job.process','model.call','POST /score','model.forward']
    rows=[next(s for s in spans if s['name']==name) for name in order];origin=rows[0]['start_ns']
    starts=[(s['start_ns']-origin)/1e6 for s in rows];durations=[(s['end_ns']-s['start_ns'])/1e6 for s in rows]
    fig,ax=plt.subplots(figsize=(13,7));fig.subplots_adjust(left=.27,right=.94,bottom=.27,top=.75)
    ys=np.arange(7);colors=[BLUE,GOLD,GOLD,GREEN,BLUE,GREEN,GREEN]
    ax.barh(ys,durations,left=starts,color=colors,height=.6)
    ax.set_yticks(ys,[s['name'] for s in rows]);ax.invert_yaxis();maximum=max(a+b for a,b in zip(starts,durations));ax.set_xlim(0,maximum*1.3)
    for y,start,duration in zip(ys,starts,durations):ax.text(start+duration+maximum*.02,y,f'{duration:.3f} ms',va='center',fontsize=9)
    ax.set_xlabel('ms từ đầu API span');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('02 · Waterfall từ một trace SDK đã chạy',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'7 span cho một request: publish/wait, consumer, HTTP model và forward. Timestamps là số đo của run này.',fontsize=10,color=GRAY)
    fig.text(.055,.13,'Delay cooperative được chèn để học; không lấy một trace làm p95/p99 hoặc benchmark model.',fontsize=11,color=GRAY)
    fig.text(.055,.06,'Không cộng mọi bar: parent/child lồng nhau; consumer có thể bắt đầu sau khi producer kết thúc.',fontsize=11,color=RED)
    save(fig,'02-trace-waterfall',out)

    fig,ax=canvas('03 · Queue phải mang context, không chỉ payload',
                  'Worker tạo trước request. W3C carrier được inject ở producer, extract khi worker nhận message.')
    for y,color,worker_label,model_label in [(4.4,GREEN,'CONSUMER · T1','MODEL · T1'),(1.6,RED,'CONSUMER · T2','MODEL · T2')]:
        box(ax,.4,y,2.8,1.4,'API + PRODUCER\nTrace T1',BLUE,10)
        box(ax,5,y,2.8,1.4,worker_label,color,11);box(ax,9.5,y,2.8,1.4,model_label,color,11)
        arrow(ax,3.3,y+.7,4.9,y+.7,color);arrow(ax,7.9,y+.7,9.4,y+.7,color)
    ax.text(4.1,6.1,'Giữ carrier: cùng trace',ha='center',fontsize=11,color=GREEN)
    ax.text(4.1,3.3,'Bỏ carrier: mở trace mới',ha='center',fontsize=11,color=RED)
    ax.text(6.5,.4,'Hai request: normal 2 trace/14 span; broken 4 trace/14 span. HTTP đều 200/200.',ha='center',fontsize=11)
    save(fig,'03-context-boundaries',out)

    fig,axes=plt.subplots(1,2,figsize=(13,7));fig.subplots_adjust(left=.1,right=.95,bottom=.27,top=.73,wspace=.35)
    values=[x*1000 for x in h['latency_seconds']]
    axes[0].bar(['A','B','C','D','E'],values,color=BLUE)
    axes[0].axhline(h['mean_seconds']*1000,color=GOLD,ls='--',label='Mean 120 ms')
    axes[0].axhline(h['p95_nearest_rank']*1000,color=RED,ls=':',label='p95 nearest-rank 500 ms')
    axes[0].set(ylabel='Latency (ms)',ylim=(0,560));axes[0].legend(fontsize=9);axes[0].set_title('Năm giá trị giả định')
    buckets=h['histogram']['cumulative'];axes[1].bar(['≤20','≤50','≤100','+Inf'],buckets,color=GREEN)
    axes[1].set(xlabel='Bucket upper bound (ms)',ylabel='Cumulative count',ylim=(0,6),yticks=range(7));axes[1].set_title('Cumulative histogram')
    for i,n in enumerate(buckets):axes[1].text(i,n+.1,str(n),ha='center')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('04 · Mean, quantile và bucket có ý nghĩa khác nhau',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'10/20/30/40/500 ms → mean 120 ms. Buckets 2/4/4/5; không là các bin rời 2/2/0/1.',fontsize=11,color=GRAY)
    fig.text(.055,.13,'Ví dụ tính tay, không lấy từ trace demo. N=5 quá nhỏ để suy tail latency production ổn định.',fontsize=11,color=GRAY)
    fig.text(.055,.06,'Không lấy trung bình p95 của replica để có p95 toàn hệ thống.',fontsize=11,color=RED)
    save(fig,'04-histograms',out)

    fig,ax=plt.subplots(figsize=(12,7));fig.subplots_adjust(left=.12,right=.95,bottom=.28,top=.74)
    x=np.arange(2);acc=[q['early']['observed_metrics']['accuracy'],q['mature']['observed_metrics']['accuracy']]
    coverage=[q['early']['label_coverage'],q['mature']['label_coverage']]
    ax.bar(x-.18,np.array(acc)*100,width=.34,color=BLUE,label='Accuracy trên phần có nhãn')
    ax.bar(x+.18,np.array(coverage)*100,width=.34,color=GOLD,label='Label coverage')
    for i,(a,c) in enumerate(zip(acc,coverage)):
        ax.text(i-.18,a*100+2,'6/6' if i==0 else '6/10',ha='center')
        ax.text(i+.18,c*100+2,'6/10' if i==0 else '10/10',ha='center')
    ax.set(xticks=x,xticklabels=['as_of=120\nPROVISIONAL','as_of=220\nREADY'],ylabel='%',ylim=(0,125));ax.legend(loc='upper center',fontsize=10)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('05 · Cùng cohort, nhãn đến muộn làm thay đổi cách hiểu',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Current cohort [100,110): 6 prediction đúng có nhãn sớm; 4 prediction sai có nhãn muộn, cố ý tạo bias.',fontsize=10,color=GRAY)
    fig.text(.055,.13,'Maturity và coverage là policy kiểm dữ liệu, không là bằng chứng missing labels random.',fontsize=11,color=GRAY)
    fig.text(.055,.06,'6/6 đúng lúc đầu không được quảng bá là 10/10. Thiếu nhãn không tự là label 0 hoặc lỗi prediction.',fontsize=11,color=RED)
    save(fig,'05-delayed-feedback',out)

    fig,axes=plt.subplots(1,2,figsize=(13,7));fig.subplots_adjust(left=.1,right=.95,bottom=.28,top=.73,wspace=.35)
    points=[(0,.8,'Reference',BLUE),(q['current_distribution']['tv'],.6,'Current',RED),
            (q['shift_distribution']['tv'],1,'Shift',GREEN)]
    for x,y,label,color in points:
        axes[0].scatter([x],[y],s=65,color=color);axes[0].annotate(label,xy=(x,y),xytext=(x+.025,y-.03),fontsize=10,color=color)
    axes[0].set(xlim=(-.04,.45),ylim=(0,1.15),xlabel='Feature histogram TV proxy',ylabel='Mature accuracy');axes[0].set_title('Drift proxy không là quality verdict')
    segs=['mobile','web'];x=np.arange(2)
    before=[q['reference']['slices'][s]['metrics']['accuracy'] for s in segs]
    after=[q['mature']['slices'][s]['metrics']['accuracy'] for s in segs]
    axes[1].bar(x-.18,before,width=.34,color=BLUE,label='Reference')
    axes[1].bar(x+.18,after,width=.34,color=RED,label='Current')
    for i,(a,b) in enumerate(zip(before,after)):
        axes[1].text(i-.18,a+.03,f'{a:.0%}',ha='center');axes[1].text(i+.18,b+.03,f'{b:.0%}',ha='center')
    axes[1].set(xticks=x,xticklabels=segs,ylim=(0,1.25),ylabel='Accuracy');axes[1].legend(loc='upper center',fontsize=10);axes[1].set_title('Mỗi slice n=5, coverage=100%')
    for ax in axes:ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('06 · Nhìn thêm outcome và slice sau distribution',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Current: TV=0 nhưng accuracy 0,8→0,6; shift: TV=0,3 nhưng accuracy=1. Counterexamples synthetic.',fontsize=10,color=GRAY)
    fig.text(.055,.13,'Hai-bin univariate proxy có thể bỏ sót joint/in-bin changes; .3 không phải p-value.',fontsize=11,color=GRAY)
    fig.text(.055,.06,'Current overall 60% che mobile 20% / web 100%; cần mẫu số và độ phủ nhãn của từng slice.',fontsize=11,color=RED)
    save(fig,'06-drift-and-slices',out)

    fig,ax=canvas('07 · Chẩn đoán bằng dữ liệu đủ và đúng ranh giới',
                  'Alert là tín hiệu điều tra; quality cần nhãn, còn trace giúp định vị path/operation.')
    box(ax,.4,4.5,3.2,1.4,'SIGNAL\nLatency / error / quality',BLUE,10)
    box(ax,4.8,4.5,3.2,1.4,'OBSERVATION PIPELINE\nSampler / exporter / join',GOLD,10)
    box(ax,9.2,4.5,3.1,1.4,'WINDOW / MẪU SỐ\nn / coverage / maturity',GOLD,10)
    arrow(ax,3.7,5.2,4.7,5.2);arrow(ax,8.1,5.2,9.1,5.2);arrow(ax,10.75,4.4,10.75,3.2)
    box(ax,9.2,1.6,3.1,1.4,'ĐỦ NHÃN\nQuality + slices',GREEN,10)
    box(ax,4.8,1.6,3.2,1.4,'REPRODUCE\nVersion / data / eval',BLUE,10)
    box(ax,.4,1.6,3.2,1.4,'HÀNH ĐỘNG CÓ DUYỆT\nSửa pipeline / rollback nếu cần',GREEN,9)
    arrow(ax,9.1,2.3,8.1,2.3);arrow(ax,4.7,2.3,3.7,2.3)
    ax.text(6.5,.45,f"Lab: {r['evaluation']['passed']}/{r['evaluation']['checks']} regression check; chỉ trả alert records, không tự gửi thông báo hoặc đổi model.",ha='center',fontsize=11)
    save(fig,'07-diagnosis',out)
    print(json.dumps(dict(figures=7,formats=['png','svg'],checks=r['evaluation']['checks'])))


if __name__=='__main__':main()
