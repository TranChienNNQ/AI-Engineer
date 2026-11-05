"""Recreate seven diagrams/charts from the offline lab report."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from lab import report

ROOT=Path(__file__).resolve().parent
BLUE='#2863a5';GREEN='#227b62';RED='#bd4949';GRAY='#596777';GOLD='#d89a24'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})


def canvas(title,subtitle,size=(13,7.5)):
    fig,ax=plt.subplots(figsize=size);ax.set(xlim=(0,13),ylim=(0,7));ax.axis('off')
    fig.suptitle(title,x=.055,y=.96,ha='left',fontsize=19,fontweight='bold',color='#172d45')
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
    r=json.loads(args.report.read_text()) if args.report else report();out=args.out;out.mkdir(parents=True,exist_ok=True)
    rows={x['name']:x for x in r['evaluation']['rows']}
    fig,ax=canvas('01 · Agent: chọn bước tiếp từ observation',
                  'Model/policy chọn action; host giữ quyền, task, budget và state. Lab dùng policy Python.')
    box(ax,.4,4.4,2.5,1.5,'OBSERVE\nState + memory\nKiểm quyền / độ mới',BLUE)
    box(ax,4,4.4,2.5,1.5,'DECIDE\nPolicy / model\nTool · final · handoff',GOLD)
    box(ax,7.6,4.4,2.5,1.5,'GUARD\nSchema + task\nReservation + giới hạn',GREEN)
    arrow(ax,3,5.1,3.9,5.1);arrow(ax,6.6,5.1,7.5,5.1)
    box(ax,7.6,1.2,2.5,1.5,'ACT\nRead tool\nKết quả hoặc lỗi',BLUE);arrow(ax,8.8,4.3,8.8,2.8)
    box(ax,3.7,1.2,2.8,1.5,'UPDATE\nObservation + budget\nCheckpoint + event',GREEN)
    arrow(ax,7.5,2,6.6,2);arrow(ax,3.6,2,1.6,4.3)
    box(ax,10.9,4.4,1.7,1.5,'STOP\nFinal gate\nhoặc giới hạn',RED,10);arrow(ax,10.2,5.1,10.8,5.1,RED)
    ax.text(6.5,.35,'Vòng lặp hữu hạn; không cần lưu suy nghĩ nội bộ để kiểm action/state/evidence.',ha='center',fontsize=12)
    save(fig,'01-agent-loop',out)

    fig,ax=canvas('02 · State, context và memory có scope khác nhau',
                  'Short-term nói về phạm vi run/thread; dữ liệu vẫn có thể nằm trong file SQLite.')
    box(ax,.4,3.4,3.3,2.4,'WORKING STATE — RUN r1\nTask + observations + refs\nPending + counters + revision\nHost sở hữu',BLUE)
    box(ax,4.9,3.4,3.2,2.4,'CONTEXT — TURN t\nPhần state còn hợp lệ\nSummary + memory chọn lọc\nCó cap hữu hạn',GOLD)
    box(ax,9.2,3.4,3.3,2.4,'LONG-TERM MEMORY\nNamespace (tenant, actor)\nRecall qua run r1 / r2\nTTL + version + origins',GREEN)
    arrow(ax,3.8,4.6,4.8,4.6);arrow(ax,9.1,4.6,8.2,4.6)
    box(ax,.4,.9,3.3,1.3,'CHECKPOINT + EVENT\nLatest state bền qua restart',GREEN);arrow(ax,2,3.3,2,2.3)
    ax.text(6.5,1.8,'Context không bằng toàn bộ DB.\nGhi memory không đổi trọng số model.',ha='center',va='center',fontsize=12)
    ax.text(6.5,.2,'Cache tránh tính lại; checkpoint khôi phục run; memory giữ dữ liệu để recall có policy.',ha='center',fontsize=11)
    save(fig,'02-state-context-memory',out)

    fig,ax=canvas('03 · Checkpoint: crash ở đâu thì biết đến đó',
                  'Reservation commit trước call; read có thể chạy lại nếu observation chưa commit.')
    box(ax,.4,4.7,3,1.2,'RESERVE TOOL\nPersist PENDING_TOOL',GOLD)
    box(ax,5,4.7,3,1.2,'READ TOOL\nNhận response',BLUE)
    box(ax,9.6,4.7,3,1.2,'COMMIT OBSERVATION\nState = READY',GREEN)
    arrow(ax,3.5,5.3,4.9,5.3);arrow(ax,8.1,5.3,9.5,5.3)
    ax.text(4.1,4,'A: crash trước read',ha='center',color=RED);ax.text(8.8,4,'B: crash trước save',ha='center',color=RED)
    values=[('Bình thường','happy'),('A: trước read','crash_before_read'),('B: sau read','crash_after_read')]
    for i,(label,key) in enumerate(values):
        row=rows[key];y=2.8-i*.8
        ax.text(.7,y,label,va='center',fontsize=12)
        ax.text(5.3,y,f"{row['tool_reservations']} reservations",va='center',ha='center',color=GOLD,fontsize=12)
        ax.text(9.8,y,f"{row['actual_reads']} actual read attempts",va='center',ha='center',color=BLUE,fontsize=12)
    ax.text(6.5,.25,'Không reset budget khi restart. Read replay ≠ exactly-once; write cần idempotency riêng.',ha='center',fontsize=12)
    save(fig,'03-checkpoint-recovery',out)

    fig,ax=canvas('04 · Working summary phải giữ phủ định và khoảng trống',
                  'Summary được tạo lại từ observation còn quyền/còn mới; chưa là LLM compaction benchmark.')
    box(ax,.4,3.4,3.5,2.4,'OBSERVATIONS\nTicket: checkout / technical\nRetry: timeout, unavailable\nKHÔNG retry: permission_denied\nRefs: ticket + KB',BLUE)
    box(ax,5.1,3.4,3.5,2.4,'STRUCTURED SUMMARY\nAcquired: ticket, runbook\nOpen: get_health\nnot_retry: permission_denied\nOrigins + last_error',GREEN)
    arrow(ax,4,4.6,5,4.6)
    box(ax,9.6,3.4,2.9,2.4,'SUMMARY SAI\n“Retry mọi lỗi”\nMất phủ định\nNguồn đúng chưa đủ',RED)
    ax.text(6.5,1.9,'Recent chỉ giữ 2 event; latest state không bằng toàn bộ lịch sử.',ha='center',fontsize=12)
    ax.text(6.5,.8,'Lab vẫn cấp cả observations + summary: không công bố tỷ lệ nén toàn context.',ha='center',color=GRAY)
    save(fig,'04-working-memory',out)

    fig,ax=canvas('05 · Memory: review, scope, hết hạn và quên',
                  'Memory hợp quyền vẫn là data; không được cấp role, sửa limit hoặc trở thành system instruction.')
    box(ax,.3,4.6,2.5,1.4,'CANDIDATE\nNội dung từ tương tác\nChưa tự là fact',GOLD)
    box(ax,3.6,4.6,2.5,1.4,'TRUSTED REVIEW\nScope + source refs\nGiả định có intent',GREEN)
    box(ax,6.9,4.6,2.5,1.4,'STORE\nKey + version\nNamespace + TTL',BLUE)
    box(ax,10.2,4.6,2.5,1.4,'RECALL\nQuyền + freshness\nLexical top-k',GREEN)
    for x in [2.9,6.2,9.5]:arrow(ax,x,5.3,x+.6,5.3)
    box(ax,.5,1.3,3.6,1.8,'NAMESPACE\nAlice / blue → được recall\nBob / blue → không chia sẻ\nCarol / green → không chia sẻ',BLUE)
    box(ax,4.7,1.3,3.6,1.8,'TTL\nCreated 10 + TTL 20 = 30\nnow 29: còn dùng\nnow 30: hết hạn',GOLD)
    box(ax,8.9,1.3,3.6,1.8,'INVALIDATION\nSource đổi / revoke → loại\nForget → tombstone\nKhông hứa xóa mọi bản cũ',RED)
    ax.text(6.5,.35,'Cross-session recall là đọc dữ liệu ngoài context; model weights không đổi.',ha='center',fontsize=12)
    save(fig,'05-memory-lifecycle',out)

    fig,axes=plt.subplots(1,2,figsize=(13,7));fig.subplots_adjust(left=.16,right=.96,bottom=.24,top=.72,wspace=.5)
    labels=['System','Task','Tool schema','Output reserve','Còn lại'];values=[128,64,96,128,608]
    axes[0].barh(labels,values,color=[BLUE,GRAY,GREEN,RED,GOLD]);axes[0].invert_yaxis();axes[0].set_title('Ví dụ tính tay: window 1.024')
    axes[0].set_xlim(0,700);axes[0].set_xlabel('Token giả định')
    for i,v in enumerate(values):axes[0].text(v+10,i,str(v),va='center')
    keys=['happy','memory_poisoned'];vals=[rows[k]['context_units'] for k in keys]
    axes[1].bar(['Không memory','Có memory độc hại'],vals,color=[GREEN,GOLD],width=.6)
    axes[1].set_title('Lab: tổng qua 4 model turn');axes[1].set_ylabel('Word units + output reserve');axes[1].set_ylim(0,max(vals)*1.25)
    for i,v in enumerate(vals):axes[1].text(i,v+20,str(v),ha='center',fontsize=12)
    for ax in axes:ax.grid(axis='x' if ax is axes[0] else 'y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('06 · Memory thêm vào context cũng tiêu ngân sách',x=.055,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.055,.86,'Hai cách đếm khác nhau: ví dụ token tính tay và tokenizer word-level custom của lab.',fontsize=11,color=GRAY)
    fig.text(.055,.1,'Không suy memory cải thiện hay làm giảm chất lượng LLM từ những số này; chưa gọi LLM thật.',fontsize=11,color=RED)
    save(fig,'06-context-budget',out)

    states=['DONE','HANDOFF','STALLED','LIMIT_REACHED','FAILED','CANCELLED'];counts=r['evaluation']['states']
    fig,ax=plt.subplots(figsize=(13,7));fig.subplots_adjust(left=.08,right=.97,bottom=.3,top=.73)
    x=np.arange(len(states));vals=[counts.get(s,0) for s in states]
    ax.bar(x,vals,color=[GREEN,BLUE,GOLD,GOLD,RED,GRAY],width=.65);ax.set_xticks(x,states,rotation=15,ha='right');ax.set_ylim(0,11)
    ax.set_ylabel('Số scenario (25 DB độc lập)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for i,v in enumerate(vals):ax.text(i,v+.2,str(v),ha='center',fontsize=13,fontweight='bold')
    fig.suptitle('07 · 25/25 expected status chưa là task accuracy',x=.055,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.055,.86,'Scripted policy + host regression · DONE chỉ qua evidence gate cơ học.',fontsize=11,color=GRAY)
    fig.text(.055,.13,'Residual: 1 trong 9 DONE trả BANANA_OVERRIDE dù source IDs hợp lệ; nội dung vẫn sai.',fontsize=12,color=RED)
    fig.text(.055,.06,'Không có benchmark LLM, memory usefulness, token billing hoặc latency mạng trong bài.',fontsize=11,color=GRAY)
    save(fig,'07-evaluation',out)
    print(json.dumps(dict(figures=7,formats=['png','svg'],cases=r['evaluation']['cases'],dataset_hash=r['dataset_hash'])))

if __name__=='__main__':main()
