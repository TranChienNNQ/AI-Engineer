"""Seven PNG/SVG teaching figures, generated from actual report and labeled arithmetic."""
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
        from lab import report
        r=report()
    out=args.out;out.mkdir(parents=True,exist_ok=True);summary=r['evaluation']['summary']
    fig,ax=canvas('01 · Đánh giá cả một lần làm việc',
                  'Task + world reset → agent hành động → journal/state → grader. Oracle ở ngoài policy view.')
    box(ax,.5,4.6,2.6,1.3,'TASK / MANIFEST\nInstruction + initial state',BLUE,10)
    box(ax,4.1,4.6,3.4,1.3,'AGENT HARNESS\nPolicy → gateway → tools',GREEN)
    box(ax,8.5,4.6,3.8,1.3,'ENVIRONMENT\nTicket / runbook / health / note',BLUE,10)
    arrow(ax,3.2,5.25,4,5.25);arrow(ax,7.6,5.25,8.4,5.25)
    box(ax,.5,1.5,2.6,1.5,'ORACLE\nLabels + protected state',GOLD,10)
    box(ax,4.1,1.5,3.4,1.5,'GRADERS\nOutcome / path / safety / caps',GOLD,10)
    box(ax,8.5,1.5,3.8,1.5,'OBSERVATIONS FOR EVAL\nFinal + journal + counters',BLUE,10)
    arrow(ax,3.2,2.2,4,2.2);arrow(ax,8.4,2.2,7.6,2.2);arrow(ax,10.4,4.5,10.4,3.1)
    ax.text(6.5,.5,'Evaluation harness bao quanh agent harness; không dùng lời tự khai thành công làm oracle.',ha='center',fontsize=11)
    save(fig,'01-evaluation-harness',out)

    fig,ax=canvas('02 · Một trial có thể đúng kết quả nhưng sai hành vi',
                  'Các điều kiện được chấm riêng; accepted = outcome ∧ path ∧ safe ∧ within_budget.')
    entries=[(.5,'OUTCOME\nNote / facts / version',BLUE),(3.7,'PATH\nRead tiền đề trước effect',GREEN),
             (6.9,'SAFETY\nScope / intent / review',GOLD),(10.1,'CAPS\nTurn / call / cost / time',BLUE)]
    for x,label,color in entries:box(ax,x,4.6,2.4,1.4,label,color,10)
    box(ax,.5,2.1,5.4,1.5,'Strict D01/profile 2\n1 proposal bị chặn; outcome đúng; safe; accepted',GREEN,11)
    box(ax,6.8,2.1,5.4,1.5,'Schema-only cùng profile\nOutcome đúng; đã đọc green; không accepted',RED,11)
    ax.text(6.5,.75,f"Adaptive / strict: {summary['accepted']}/{summary['trials']} accepted; "
            f"{summary['safe_trials']}/{summary['trials']} safe; {r['checks']['passed']}/{r['checks']['checks']} regression check.",ha='center',fontsize=12)
    ax.text(6.5,.2,'Mỗi số có mẫu số và ý nghĩa riêng. Chưa có model/production benchmark.',ha='center',fontsize=11,color=GRAY)
    save(fig,'02-grading-layers',out)

    fig,ax=canvas('03 · Giữ phụ thuộc, cho phép nhiều thứ tự hợp lệ',
                  'Không ép thứ tự giữa ba read; phải có đủ evidence trước prepare/review/commit.')
    box(ax,.4,3,2.4,1.3,'GET TICKET',BLUE)
    box(ax,.4,5.1,2.4,1.2,'GET RUNBOOK',BLUE)
    box(ax,.4,.9,2.4,1.2,'GET HEALTH',BLUE)
    box(ax,7,3,2.1,1.3,'PREPARE\n+ REVIEW',GOLD)
    box(ax,10.2,3,2.2,1.3,'COMMIT\n→ FINAL NOTE',GREEN)
    arrow(ax,2.9,3.65,6.9,3.65);arrow(ax,2.9,5.7,6.9,3.95)
    arrow(ax,2.9,1.5,6.9,3.35);arrow(ax,9.2,3.65,10.1,3.65)
    ax.text(6.5,.55,'Chấm theo journal + điều kiện, không so transcript với duy nhất một đường mẫu.',ha='center',fontsize=12)
    save(fig,'03-partial-order',out)

    fig,ax=canvas('04 · Reserve trước dispatch, kiểm lại trước effect',
                  'Giới hạn nằm trong trusted host; output model không được sửa counter hoặc tự cấp review.')
    box(ax,.5,4.1,2.8,1.6,'MODEL TURN\nView cap → reserve\n→ parse decision',BLUE)
    box(ax,4.7,4.1,3.1,1.6,'TOOL BOUNDARY\nReserve → scope/intent\n→ evidence/review/version',GOLD,10)
    box(ax,9.2,4.1,2.8,1.6,'DOMAIN EFFECT\nRead / commit note\n→ journal + result',GREEN)
    arrow(ax,3.4,4.9,4.6,4.9);arrow(ax,7.9,4.9,9.1,4.9)
    box(ax,.5,1.3,5.3,1.6,'Happy: 6 turn + 5 tool call\n19 synthetic units / 310 modeled ms',BLUE)
    box(ax,6.8,1.3,5.3,1.6,'Deadline 300 ms\nWrite đã xong; final bị chặn; cần đối soát',RED)
    ax.text(6.5,.35,'Đồng hồ/cost giả, tuần tự trong RAM; chưa actual I/O timeout, billing hoặc distributed quota.',ha='center',fontsize=11)
    save(fig,'04-action-boundary',out)

    h=r['hand'];ks=np.arange(1,5)
    fig,ax=plt.subplots(figsize=(12,7));fig.subplots_adjust(left=.1,right=.95,bottom=.25,top=.75)
    ax.plot(ks,[h['metrics'][str(k)]['at_least_one'] for k in ks],'-o',color=BLUE,lw=2,label='Ít nhất một lần thành công')
    ax.plot(ks,[h['metrics'][str(k)]['all_success'] for k in ks],'-o',color=GREEN,lw=2,label='Tất cả đều thành công')
    ax.set(xticks=ks,xlabel='k trial được chọn từ 4 trial/task',ylabel='Macro score theo task',ylim=(0,1.08));ax.grid(alpha=.2)
    ax.legend(loc='center right');ax.annotate('k=2: 7/9 ≈ 0,778',xy=(2,7/9),xytext=(2.5,.91),arrowprops={'arrowstyle':'->','color':BLUE})
    ax.annotate('k=2: 7/18 ≈ 0,389',xy=(2,7/18),xytext=(2.6,.25),arrowprops={'arrowstyle':'->','color':GREEN})
    fig.suptitle('05 · pass@k và pass^k khác câu hỏi',x=.055,y=.95,ha='left',fontsize=19,fontweight='bold')
    fig.text(.055,.85,'Ví dụ tính tay A=1111, B=1100, C=1000. Liệt kê subset; không lấy p trung bình rồi lũy thừa.',fontsize=10,color=GRAY)
    fig.text(.055,.13,'Với trial i.i.d., công thức có diễn giải estimator reliability; với profile cố định, chỉ là mô tả subset.',fontsize=10,color=GRAY)
    fig.text(.055,.06,'pass@k cao không tự cho phép retry k write trên dữ liệu production.',fontsize=11,color=RED)
    save(fig,'05-repeat-metrics',out)

    rows=r['comparison']['results'];labels=[x['policy']+' / '+x['mode'] for x in rows];ys=np.arange(len(rows))
    fig,axes=plt.subplots(1,3,figsize=(15,7),sharey=True);fig.subplots_adjust(left=.24,right=.96,bottom=.22,top=.73,wspace=.25)
    for ax,key,den,color,title in zip(axes,['accepted','positive_completed','violation_trials'],[48,24,48],
                                   [GREEN,BLUE,RED],['Accepted / 48','Positive hoàn thành / 24','Trial có vi phạm / 48']):
        vals=[x[key]/den*100 for x in rows];ax.barh(ys,vals,color=color,height=.65)
        ax.set(xlim=(0,115),xticks=[0,50,100],xlabel='%');ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for y,x,val in zip(ys,rows,vals):ax.text(val+2,y,str(x[key])+'/'+str(den),va='center',fontsize=9)
    axes[0].set_yticks(ys,labels,fontsize=10);axes[0].invert_yaxis()
    fig.suptitle('06 · An toàn, hữu ích và từ chối cần được báo riêng',x=.055,y=.95,ha='left',fontsize=19,fontweight='bold')
    fig.text(.055,.85,'Cùng 12 task × 4 profile, world reset cho mỗi lượt. Schema-only cố ý bỏ domain guards.',fontsize=11,color=GRAY)
    fig.text(.055,.1,'Refuse: 0/24 positive, nhưng 24/48 accepted nhờ nhóm expected HANDOFF. Chưa chấm lý do handoff.',fontsize=11,color=GRAY)
    save(fig,'06-baselines',out)

    from evaluation import zero_event_upper
    fig,axes=plt.subplots(1,2,figsize=(13,7));fig.subplots_adjust(left=.1,right=.95,bottom=.27,top=.74,wspace=.4)
    lo,hi=h['wilson_7_of_12'];p0=7/12
    axes[0].errorbar([p0],[0],xerr=[[p0-lo],[hi-p0]],fmt='o',capsize=10,color=BLUE,markersize=9)
    axes[0].set(xlim=(0,1),ylim=(-1,1),yticks=[],xlabel='Tỷ lệ thành công');axes[0].set_title('7/12 — Wilson xấp xỉ 95%',fontsize=12)
    axes[0].text(.5,.4,'[0,3195; 0,8067]',ha='center',fontsize=13,color=BLUE);axes[0].grid(axis='x',alpha=.2)
    ns=np.arange(5,151);axes[1].plot(ns,[zero_event_upper(int(n)) for n in ns],color=RED,lw=2)
    axes[1].scatter([30],[zero_event_upper(30)],color=RED)
    axes[1].annotate('0/30 sự kiện\nUpper ≈ 9,50%',xy=(30,zero_event_upper(30)),xytext=(60,.25),arrowprops={'arrowstyle':'->','color':RED})
    axes[1].set(xlabel='n trial i.i.d. không thấy sự kiện',ylabel='Upper bound một phía 95%',ylim=(0,.5));axes[1].grid(alpha=.2)
    fig.suptitle('07 · Không quan sát thấy lỗi chưa chứng minh rủi ro bằng 0',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Hai ví dụ nhị thức i.i.d. giả định; không áp interval lên 41 check hoặc 48 profile lập trình sẵn.',fontsize=10,color=GRAY)
    fig.text(.055,.12,'Wilson: hai phía, z=1,96. Không sự kiện: một phía, q_U = 1 − 0,05^(1/n).',fontsize=11,color=GRAY)
    fig.text(.055,.06,'Trial cùng task có thể tương quan; cần thiết kế lấy mẫu/nhóm phù hợp trước suy luận thống kê.',fontsize=11,color=GRAY)
    save(fig,'07-uncertainty',out)
    print(json.dumps(dict(figures=7,formats=['png','svg'],trials=summary['trials'],accepted=summary['accepted'])))


if __name__=='__main__':main()
