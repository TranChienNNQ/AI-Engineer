"""Seven standalone teaching figures from executed simulation/hand/cost definitions."""
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


def canvas(title,subtitle):
    fig,ax=plt.subplots(figsize=(13,7.5));ax.set(xlim=(0,13),ylim=(0,7));ax.axis('off')
    fig.suptitle(title,x=.055,y=.96,ha='left',fontsize=18,fontweight='bold',color='#172d45')
    fig.text(.055,.89,subtitle,fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.04,right=.98,bottom=.08,top=.84)
    return fig,ax


def box(ax,x,y,w,h,label,color=BLUE,fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.12',
                              facecolor=color+'12',edgecolor=color,linewidth=1.5))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color='#172d45',linespacing=1.6)


def arrow(ax,x,y,xx,yy,color=GRAY):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=15,color=color,lw=1.5))


def save(fig,name,out):
    for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=170,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'figures');args=p.parse_args()
    r=json.loads(args.report.read_text());out=args.out;out.mkdir(parents=True,exist_ok=True)
    fig,ax=canvas('01 · Cùng một request, nhiều điểm đặt đồng hồ',
                  'Định nghĩa của bài: client latency gồm pool/HTTP/parsing/validation; scheduled latency có thêm dispatch lag.')
    labels=[('PLANNED','Lịch muốn phát'),('SENT','Bắt đầu HTTP client'),('SERVER','Queue → slot → run'),('DONE','Response checked / error')]
    for i,(a,b) in enumerate(labels):
        x=.2+i*3.25;box(ax,x,4.25,2.8,1.3,a+'\n'+b,BLUE if i<2 else GREEN,10)
        if i<3:arrow(ax,x+2.85,4.9,x+3.17,4.9)
    box(ax,.65,2.1,4,1.1,'dispatch lag = sent − planned',GOLD,11)
    box(ax,5.6,2.1,6.65,1.1,'client latency = done − sent',BLUE,11)
    box(ax,.65,.55,11.6,.9,'scheduled latency = done − planned = dispatch lag + client latency',GREEN,11)
    save(fig,'01-measurement-boundaries',out)

    fig,axes=plt.subplots(1,2,figsize=(13,7.5));fig.subplots_adjust(left=.09,right=.96,bottom=.27,top=.75,wspace=.32)
    ax=axes[0];values=r['hand']['latencies_ms'];ax.bar(range(5),values,color=BLUE)
    ax.axhline(120,color=GOLD,ls='--',label='Mean 120 ms');ax.axhline(500,color=RED,ls=':',label='p95/p99 = 500 ms')
    ax.set_xticks(range(5),list('ABCDE'));ax.set_ylabel('Latency (ms)');ax.set_ylim(0,600);ax.legend(fontsize=9,loc='upper left');ax.set_title('Năm sample giả định, n=5')
    ax=axes[1];ax.bar(['Success only\nn=4','Errors only\nn=4','Pooled\nn=8'],[30,0,15],color=[GREEN,RED,GRAY])
    for x,y in enumerate([30,0,15]):ax.text(x,y+1.2,f'{y} ms',ha='center')
    ax.set_ylim(0,42);ax.set_ylabel('Mean latency (ms)');ax.set_title('Burst model: fast503 làm mean giảm')
    fig.suptitle('02 · Giữ population và count khi đọc latency',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Trái: ví dụ tính tay. Phải: simulation2 worker/Q2, 8 arrivals cùng lúc, success20/20/40/40 và lỗi0.',fontsize=10,color=GRAY)
    fig.text(.055,.13,'Rejected latency0 là giả định admission tức thời của mô phỏng; HTTP thật có overhead.',fontsize=11,color=RED)
    fig.text(.055,.06,'N nhỏ: p99 gần/đúng maximum, chưa chứng minh đuôi production.',fontsize=11,color=GRAY)
    save(fig,'02-latency-populations',out)

    fig,ax=plt.subplots(figsize=(13,7.5));fig.subplots_adjust(left=.13,right=.96,bottom=.24,top=.75)
    closed=r['simulation']['closed_one']['rows'];opened=r['simulation']['open_same_demand']['rows']
    for rows,y,color,label in [(closed,1.45,BLUE,'Closed: chờ xong mới gửi'),(opened,.45,RED,'Open: arrival mỗi20 ms')]:
        for i,row in enumerate(rows):
            start=row['planned_ms'];end=row['done_ms'];yy=y+(i-4.5)*.065
            ax.plot([start,end],[yy,yy],color=color,alpha=.7,lw=2)
            ax.scatter(start,yy,color=color,s=16);ax.scatter(end,yy,color=color,s=25,marker='|')
        ax.text(1030,y,label,fontsize=10,color=color,va='center')
    ax.set(xlim=(-20,1470),ylim=(-.1,2),xlabel='Modeled ms; dot=arrival, tick=completion')
    ax.set_yticks([.45,1.45],['Open','Closed']);ax.grid(axis='x',alpha=.15)
    fig.suptitle('03 · Arrival process làm thay đổi latency quan sát',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Một worker, service100 ms, queue đủ lớn, 10 request. Đây là timeline mô phỏng, không wall-clock trace.',fontsize=10,color=GRAY)
    fig.text(.055,.12,'Closed p95=100 ms. Open p95=820 ms: nhu cầu vẫn tới trong khi request trước đang xử lý.',fontsize=11,color=RED)
    fig.text(.055,.055,'Closed không luôn sai; chọn model tải theo hành vi người dùng/sự kiện cần mô phỏng.',fontsize=11,color=GRAY)
    save(fig,'03-open-closed',out)

    sweep=r['simulation']['sweep'];rates=[20,60,100,140,200]
    fig,axes=plt.subplots(1,2,figsize=(13,7.5));fig.subplots_adjust(left=.08,right=.96,bottom=.28,top=.75,wspace=.3)
    ax=axes[0];ax.plot(rates,[sweep[str(x)]['summary']['success_rps'] for x in rates],'o-',color=BLUE,label='Success/window')
    ax.plot(rates,[sweep[str(x)]['summary']['goodput_rps'] for x in rates],'o-',color=GREEN,label='Good≤40 ms/window')
    ax.set(xlabel='Intended arrivals/s',ylabel='Requests/s, gồm drain');ax.legend(fontsize=9);ax.grid(alpha=.2)
    ax=axes[1];positions=np.arange(5)
    good=[sweep[str(x)]['summary']['good'] for x in rates]
    late=[sweep[str(x)]['summary']['successful']-g for x,g in zip(rates,good)]
    rejected=[sweep[str(x)]['summary']['errors'] for x in rates]
    ax.bar(positions,good,color=GREEN,label='Good');ax.bar(positions,late,bottom=good,color=GOLD,label='Success late')
    ax.bar(positions,rejected,bottom=np.array(good)+late,color=RED,label='503');ax.set_xticks(positions,rates)
    ax.set(xlabel='Intended arrivals/s',ylabel='Count / 40 planned',ylim=(0,50));ax.legend(fontsize=9,ncol=3,loc='upper center')
    fig.suptitle('04 · Raw throughput có thể đứng, goodput vẫn giảm',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Exact-time simulation:2 worker, service20 ms, Q4, SLO40 ms; không dùng làm capacity của HTTP/GPU.',fontsize=10,color=GRAY)
    fig.text(.055,.13,'140/s:32 success nhưng chỉ8 good. 200/s:24 success,6 good. Báo cả lỗi và mẫu số.',fontsize=11,color=RED)
    fig.text(.055,.06,'Rate của cohort hữu hạn gồm fill/drain; không phải ước lượng steady-state throughput.',fontsize=11,color=GRAY)
    save(fig,'04-simulation-load-curve',out)

    fig,ax=canvas('05 · TTFT, token gaps và response end khác nhau',
                  'Timestamp token giả định, không phải kết quả đo streaming LLM hoặc SSE.')
    points=[0,120,150,190,220,240];labels=['SEND','TOKEN1','TOKEN2','TOKEN3','TOKEN4','DONE']
    xs=[.7+t/240*11.6 for t in points]
    ax.plot([.7,12.3],[4.55,4.55],color=GRAY,lw=2)
    for x,t,label in zip(xs,points,labels):
        ax.scatter(x,4.55,s=60,color=GREEN if label.startswith('TOKEN') else BLUE)
        ax.text(x,5.0,label,ha='center',fontsize=10);ax.text(x,4.0,f'{t} ms',ha='center',fontsize=10)
    box(ax,.5,1.9,3.4,1.1,'TTFT = 120 ms\nITL = 30 / 40 / 30 ms',BLUE,10)
    box(ax,4.6,1.9,3.7,1.1,'To-last-token TPOT\n100/3 ≈ 33.33 ms/token',GREEN,10)
    box(ax,9,1.9,3.5,1.1,'Response-end TPOT\n120/3 = 40 ms/token',GOLD,10)
    ax.text(6.5,.65,'Finish overhead20 ms giải thích chênh lệch. Chunk có thể chứa nhiều token; n=1 cần N/A.',ha='center',fontsize=11)
    save(fig,'05-streaming-boundaries',out)

    c=r['cost']['provider_ledger'];fig,ax=plt.subplots(figsize=(13,7.5));fig.subplots_adjust(left=.11,right=.96,bottom=.28,top=.75)
    values=[float(c['cost_per_attempt_usd']),float(c['cost_per_successful_task_usd']),float(c['cost_per_qualified_task_usd'])]
    ax.bar(range(3),values,color=[BLUE,GOLD,GREEN],width=.6)
    for i,v in enumerate(values):ax.text(i,v+.00035,f'{v:.6f} USD',ha='center',fontsize=11)
    ax.set_xticks(range(3),['Per attempt\n3 attempts','Per execution-success task\n2 tasks','Per qualified task\n1 task'])
    ax.set(ylabel='USD / đơn vị tương ứng',ylim=(0,.014));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('06 · Cùng tổng chi phí, mẫu số quyết định unit cost',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Rate card và usage đều giả định. Tổng.01055 USD đã tính failed attemptA1 và retryA2.',fontsize=10,color=GRAY)
    fig.text(.055,.13,'Qualified là annotation fixture; chưa là đánh giá model. Thiếu usage → total/per-unitcost = None.',fontsize=11,color=RED)
    fig.text(.055,.06,'Không là bảng giá hiện hành, invoice hoặc đề xuất chọn provider/GPU.',fontsize=11,color=GRAY)
    save(fig,'06-cost-denominators',out)

    fig,ax=canvas('07 · Viết protocol trước, so kết quả cùng ràng buộc',
                  'Giữ source/release, workload, clocks, window và cost scope để người khác hiểu phép đo.')
    top=[('PIN WORKLOAD','Release / mix / cache'),('PLAN LOAD','Open/closed / caps'),('MEASURE','Warmup / window / drain')]
    bottom=[('RELEASE + OBSERVE','CI / rollback / feedback'),('COMPARE','Goodput / cost / uncertainty'),('QUALIFY','Schema / errors / quality / SLO')]
    for i,(a,b) in enumerate(top):box(ax,.45+4.4*i,4.4,3.3,1.2,a+'\n'+b,BLUE,10)
    for i,(a,b) in enumerate(bottom):box(ax,.45+4.4*i,1.7,3.3,1.2,a+'\n'+b,GREEN,10)
    arrow(ax,3.85,5,4.75,5);arrow(ax,8.25,5,9.15,5);arrow(ax,10.9,4.3,10.9,3)
    arrow(ax,9.15,2.3,8.25,2.3);arrow(ax,4.75,2.3,3.85,2.3)
    ax.text(6.5,.45,f"Lab: {r['evaluation']['passed']}/{r['evaluation']['checks']} regression checks; TCP timings/counts là số đo biến đổi theo run.",ha='center',fontsize=11)
    save(fig,'07-experiment-protocol',out)


if __name__=='__main__':main()
