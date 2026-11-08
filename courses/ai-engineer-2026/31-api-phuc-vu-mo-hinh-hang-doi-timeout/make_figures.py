"""Seven diagrams/charts: report counters and explicitly labeled arithmetic, not load benchmarks."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Patch

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
    out=args.out;out.mkdir(parents=True,exist_ok=True);s=r['scenarios'];h=r['hand']
    fig,ax=canvas('01 · Một model dùng chung, nhiều request cần điều phối',
                  'Lifespan load/warm-up trước ready. API sync response và job 202 dùng cùng bounded service.')
    for x,w,label,color in [(.4,2.4,'HTTP API\nValidate input',BLUE),(3.7,2.4,'ADMISSION\nReady / capacity',GOLD),
                            (7,2.1,'FIFO QUEUE\nQ entry chờ',BLUE),(10.1,2.3,'c WORKER\nModel đã load',GREEN)]:
        box(ax,x,4.3,w,1.5,label,color,10)
    arrow(ax,2.9,5.05,3.6,5.05);arrow(ax,6.2,5.05,6.9,5.05);arrow(ax,9.2,5.05,10,5.05)
    box(ax,.4,1.4,3.3,1.5,'PREDICT\nGiữ request, await terminal',BLUE,10)
    box(ax,4.8,1.4,3.3,1.5,'JOB API\n202 + Location → poll GET',BLUE,10)
    box(ax,9.2,1.4,3.2,1.5,'RESULT / ERROR\nState + version + hash',GREEN,10)
    ax.text(6.5,.4,'Queue/job record trong RAM không là broker bền; async def không tự làm CPU inference non-blocking.',ha='center',fontsize=11)
    save(fig,'01-serving-path',out)

    fig,ax=plt.subplots(figsize=(13,7));fig.subplots_adjust(left=.13,right=.95,bottom=.28,top=.75)
    rows=h['fifo_burst']
    for y,item in enumerate(rows):
        ax.barh(y,item['wait'],left=item['arrival'],height=.5,color=GOLD,alpha=.7)
        ax.barh(y,item['finish']-item['start'],left=item['start'],height=.5,color=BLUE)
        ax.scatter([item['arrival']],[y],color=GRAY,zorder=3)
        ax.text(item['start']+50,y,'100 ms',ha='center',va='center',color='white')
        if item['wait']:ax.text((item['arrival']+item['start'])/2,y+.35,'Chờ '+str(item['wait'])+' ms',ha='center',fontsize=10,color=GRAY)
    ax.set(yticks=np.arange(3),yticklabels=['A','B','C'],xlabel='Thời gian (ms)',xlim=(-5,320),ylim=(-.6,2.8));ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.legend(handles=[Patch(color=GOLD,alpha=.7,label='Queue wait'),Patch(color=BLUE,label='Service time')],loc='upper right')
    fig.suptitle('02 · Queue hấp thụ burst, không tăng công suất worker',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Tính tay: c=1, Q=2, arrival 0/10/20/30/40 ms, service cố định 100 ms/job, chưa timeout.',fontsize=10,color=GRAY)
    fig.text(.055,.13,'D đến mốc 30 ms và E đến mốc 40 ms bị từ chối: B/C đang chiếm hai chỗ chờ.',fontsize=11,color=RED)
    fig.text(.055,.06,'Đây là lịch giả định; không là latency đo được từ sleep hoặc throughput của CPU/GPU.',fontsize=11,color=GRAY)
    save(fig,'02-fifo-burst',out)

    fig,ax=plt.subplots(figsize=(12,7));fig.subplots_adjust(left=.12,right=.95,bottom=.27,top=.75)
    arrivals=np.linspace(0,9.7,300);ax.plot(arrivals,1/(10-arrivals),color=BLUE,lw=2)
    for item in h['mm1']:
        x,y=item['arrival_rps'],item['mean_s'];ax.scatter([x],[y],color=GOLD,zorder=3)
        ax.annotate(f'{x:g} RPS → {y:g} s',xy=(x,y),xytext=(max(.5,x-3),y+.2),arrowprops={'arrowstyle':'->','color':GRAY},fontsize=10)
    ax.set(xlabel='Arrival λ (request/s)',ylabel='Mean W trong hệ thống (s)',xlim=(0,10),ylim=(0,3.6));ax.grid(alpha=.2)
    fig.suptitle('03 · Gần bão hòa, latency trung bình tăng phi tuyến',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'M/M/1 lý thuyết: μ=10/s, λ<μ; W=1/(μ−λ). Little: L=λ_admitted × W.',fontsize=11,color=GRAY)
    fig.text(.055,.13,'Giả định Poisson arrival, exponential service, một server, queue vô hạn, steady state.',fontsize=11,color=GRAY)
    fig.text(.055,.06,'Lab có queue hữu hạn, deadline và fake delay; đường cong này không phải benchmark lab.',fontsize=11,color=RED)
    save(fig,'03-utilization',out)

    fig,ax=plt.subplots(figsize=(13,7));fig.subplots_adjust(left=.12,right=.95,bottom=.27,top=.75)
    parts=[(0,120,GOLD,'Đã tiêu\n120 ms'),(120,350,BLUE,'Phần việc còn lại\n350 ms'),(470,30,GREEN,'Response\n30 ms')]
    for left,width,color,label in parts:
        ax.barh(1,width,left=left,height=.55,color=color)
        if width>50:ax.text(left+width/2,1,label,ha='center',va='center',color='white',fontsize=11)
    ax.annotate('30 ms',xy=(485,1.28),xytext=(540,1.52),arrowprops={'arrowstyle':'->','color':GREEN},color=GREEN)
    ax.barh(0,500,left=0,height=.55,color=GOLD,alpha=.7);ax.barh(0,500,left=500,height=.55,color=RED,alpha=.7)
    ax.text(250,0,'Queue cấp 500 ms',ha='center',va='center');ax.text(750,0,'Run lại được cấp 500 ms',ha='center',va='center',color='white')
    ax.axvline(500,color=GRAY,ls='--');ax.set(yticks=[0,1],yticklabels=['Reset sai','Một budget'],xlabel='ms từ đầu operation giả định',xlim=(0,1050),ylim=(-.6,1.8));ax.grid(axis='x',alpha=.15)
    fig.suptitle('04 · Giữ deadline cũ, truyền ngân sách còn lại',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Ví dụ client tổng 500 ms: 500 − 120 − 30 = 350 ms. Không cấp mới toàn budget ở mỗi stage.',fontsize=11,color=GRAY)
    fig.text(.055,.13,'Trong lab, budget bắt đầu sau validation; deadline queue/run dùng cùng monotonic clock.',fontsize=11,color=GRAY)
    fig.text(.055,.06,'Ví dụ phân bổ client ở hình không là số đo các stage hoặc policy tự động reserve response của lab.',fontsize=10,color=GRAY)
    save(fig,'04-deadline-budget',out)

    fig,ax=canvas('05 · Terminal logic khác thời điểm giải phóng tài nguyên',
                  'QUEUED/RUNNING có thể hết hạn hoặc bị hủy; kết quả tới muộn không ghi đè terminal state.')
    box(ax,.5,4.4,2.5,1.3,'QUEUED',BLUE);box(ax,4.8,4.4,2.5,1.3,'RUNNING',BLUE)
    box(ax,9.5,5.2,2.6,1.1,'SUCCEEDED',GREEN);box(ax,9.5,3.4,2.6,1.1,'FAILED',RED)
    box(ax,2.5,1.5,5.2,1.6,'EXPIRED / CANCELLED\nFuture terminal; chờ worker cleanup nếu cần',GOLD,10)
    arrow(ax,3.1,5.05,4.7,5.05);arrow(ax,7.4,5.3,9.4,5.75);arrow(ax,7.4,4.7,9.4,3.95)
    arrow(ax,1.75,4.3,3.5,3.2);arrow(ax,6.05,4.3,6.05,3.2)
    ax.text(6.5,.5,'Queue tombstone chỉ biến mất khi dequeue; busy/credit chỉ trả sau physical completion/unwind.',ha='center',fontsize=11)
    save(fig,'05-job-lifecycle',out)

    fig,ax=canvas('06 · Caller hết chờ, thread chưa chắc đã dừng',
                  'Thread demo thật: gate còn đóng ở mốc caller timeout; sau đó mở gate và await completion.')
    box(ax,.6,4.5,3.0,1.2,'CALLER\nAwait shield(task)',BLUE)
    box(ax,5.2,4.5,3.0,1.2,'TIMEOUT\nCaller ngừng đợi',RED)
    box(ax,.6,1.6,3.0,1.4,'THREAD\nFunction chờ Event',BLUE)
    box(ax,5.2,1.6,3.0,1.4,'VẪN CHƯA XONG\nTask vẫn được theo dõi',GOLD,10)
    box(ax,9.5,1.6,2.8,1.4,'MỞ GATE\nThread hoàn tất',GREEN)
    arrow(ax,3.7,5.1,5.1,5.1);arrow(ax,3.7,2.3,5.1,2.3);arrow(ax,8.3,2.3,9.4,2.3)
    t=s['thread']['at_timeout']
    ax.text(6.5,.55,f"Quan sát: caller_timed_out={t['caller_timed_out']}, thread_finished={t['thread_finished']}. Shield không kill thread.",ha='center',fontsize=11)
    save(fig,'06-cancellation',out)

    fig,axes=plt.subplots(1,2,figsize=(13,7));fig.subplots_adjust(left=.1,right=.95,bottom=.28,top=.73,wspace=.35)
    o=s['overload']['after']['counts'];v=[o['attempts'],o['accepted'],o['rejected']]
    axes[0].bar(['Attempts','Admitted','Rejected'],v,color=[BLUE,GREEN,RED]);axes[0].set_title('Burst thực: c=1, Q=1');axes[0].set_ylim(0,4)
    for i,n in enumerate(v):axes[0].text(i,n+.1,str(n),ha='center')
    e=s['expired_running'];v=[len(e['started']),len(e['finished']),len(e['cancelled'])]
    axes[1].bar(['Started','Finished','Cancelled'],v,color=[BLUE,GREEN,GOLD]);axes[1].set_title('Running deadline: backend cooperative');axes[1].set_ylim(0,2)
    for i,n in enumerate(v):axes[1].text(i,n+.05,str(n),ha='center')
    for ax in axes:ax.set_ylabel('Số job/attempt đã quan sát');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('07 · Regression đo hành vi, chưa đo tải hoặc accuracy',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,f"{r['evaluation']['passed']}/{r['evaluation']['checks']} check; {len(s)} scenario. HTTP-ASGI có queue/timer thật, delay được chèn để học.",fontsize=11,color=GRAY)
    fig.text(.055,.13,'TCP loopback kiểm riêng bằng --mode socket; không có TLS/proxy hoặc workload production.',fontsize=11,color=GRAY)
    fig.text(.055,.06,'Không suy throughput/p95/p99, model accuracy hoặc khả năng cancel GPU từ các cột đếm này.',fontsize=11,color=RED)
    save(fig,'07-verification',out)
    print(json.dumps(dict(figures=7,formats=['png','svg'],checks=r['evaluation']['checks'],scenarios=len(s))))


if __name__=='__main__':main()
