"""Six course figures, reproducible from lab report; diagrams and timings are teaching models."""
import argparse
from collections import Counter
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
    fig,ax=canvas('01 · Model đề xuất; ứng dụng thực thi',
                  'Round trip của client tool · protocol trong bài là ví dụ trung lập, không phải API provider.')
    for x,label,color in [(0.3,'MODEL\nTask + tool contract\nSinh call / nhận result',BLUE),
                           (4.8,'HOST / RUNTIME\nParse + quyền + intent\nBudget + dispatch',GREEN),
                           (9.3,'BACKEND TOOL\nĐọc / ghi theo contract\nTrả data hoặc error',BLUE)]:box(ax,x,3.4,3.2,2.1,label,color)
    arrow(ax,3.6,4.8,4.7,4.8);arrow(ax,8.1,4.8,9.2,4.8)
    arrow(ax,9.2,3.9,8.1,3.9);arrow(ax,4.7,3.9,3.6,3.9)
    ax.text(4.15,5.2,'call',ha='center',fontsize=10);ax.text(8.65,5.2,'invoke',ha='center',fontsize=10)
    ax.text(4.15,3,'result',ha='center',fontsize=10);ax.text(8.65,3,'data / error',ha='center',fontsize=10)
    box(ax,.8,.9,3.2,1.2,'call_id\nGhép call ↔ result',BLUE)
    box(ax,4.9,.9,3.2,1.2,'run_id\nNhóm các bước của workflow',GOLD)
    box(ax,9,.9,3.2,1.2,'idempotency key\nNhận diện cùng ý định write',GREEN)
    ax.text(6.5,.2,'Call hợp schema chưa có nghĩa được cấp quyền hoặc đã tạo effect.',ha='center',fontsize=12)
    save(fig,'01-tool-round-trip',out)

    fig,ax=canvas('02 · Ghép workflow theo phụ thuộc dữ liệu',
                  'Sơ đồ nghiệp vụ · workflow chính chạy tuần tự; fan-out đọc có demo asyncio riêng.')
    box(ax,.2,3.5,2,1.2,'GET TICKET\nservice + category',BLUE)
    box(ax,3.1,3.5,1.9,1.2,'ROUTE\ntechnical?',GOLD)
    arrow(ax,2.3,4.1,3,4.1)
    box(ax,3.1,.9,1.9,1.2,'HUMAN\nbilling / unknown',RED);arrow(ax,4,3.4,4,2.2,RED)
    box(ax,6,5,2,1.2,'RUNBOOK\nBắt buộc',BLUE);box(ax,6,2.9,2,1.2,'HEALTH\nCó thể unknown',BLUE)
    arrow(ax,5.1,4.4,5.9,5.5);arrow(ax,5.1,3.9,5.9,3.5)
    box(ax,9,3.8,1.6,1.5,'DRAFT\nGhép evidence',BLUE,10)
    arrow(ax,8.1,5.5,8.9,4.8);arrow(ax,8.1,3.5,8.9,4.3)
    box(ax,11.3,3.8,1.4,1.5,'GATE\nReview\n+ write',GREEN,10);arrow(ax,10.7,4.5,11.2,4.5)
    ax.text(7,1.7,'Hai read độc lập\nsau khi biết service',ha='center',color=GRAY)
    ax.text(6.5,.25,'Gate và route do code kiểm; tool output không tự đổi graph hoặc cấp quyền.',ha='center',fontsize=12)
    save(fig,'02-workflow-patterns',out)

    fig,ax=plt.subplots(figsize=(13,6.5));fig.subplots_adjust(left=.15,right=.96,bottom=.25,top=.74)
    colors=[BLUE,GOLD,GREEN,GRAY,RED]
    specs=[(2.2,[0,40,120,240,290]),(.9,[0,40,40,160,210])]
    labels=['Ticket','Runbook','Health','Draft','Write'];durations=[40,80,120,50,60]
    for row,starts in specs:
        for i,(start,duration) in enumerate(zip(starts,durations)):
            y=row
            if row==.9 and i==1:y=1.15
            if row==.9 and i==2:y=.65
            ax.barh(y,duration,left=start,height=.4,color=colors[i])
            ax.text(start+duration/2,y,f'{labels[i]}\n{duration}',ha='center',va='center',fontsize=9,color='white')
    ax.set_yticks([2.2,.9],['Tuần tự','Fan-out reads']);ax.set_xlim(0,370);ax.set_ylim(.1,2.9)
    ax.set_xlabel('Thời gian giả định (ms); chưa có thời gian chờ reviewer');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('03 · Đường găng 350 → 270; tool work vẫn 300',x=.055,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.055,.86,'Tính tay từ durations giả định · không phải latency đo qua API hoặc benchmark service.',fontsize=11,color=GRAY)
    fig.text(.055,.1,'Fan-out không giảm số request; hai write cùng tài nguyên có thể cần thứ tự và kiểm version.',fontsize=11)
    save(fig,'03-critical-path',out)

    fig,ax=canvas('04 · State machine giữ cả chờ review và UNKNOWN',
                  'Các state là tri thức của controller; UNKNOWN không tự là “chưa có side effect”.')
    positions=[(.2,'NEW',BLUE),(2.4,'READING',BLUE),(4.6,'WAITING\nREVIEW',GOLD),(6.8,'APPROVED',GREEN),(9,'WRITING',BLUE),(11.2,'SUCCEEDED',GREEN)]
    for x,label,color in positions:box(ax,x,4.6,1.6,1.15,label,color,10)
    for i in range(5):arrow(ax,positions[i][0]+1.7,5.2,positions[i+1][0]-.1,5.2)
    box(ax,1.9,1.5,2.6,1.3,'NEEDS_HUMAN\nhoặc FAILED',RED);arrow(ax,3.2,4.5,3.2,2.9,RED)
    box(ax,4.9,1.5,2,1.3,'CANCELLED\nReview từ chối',RED);arrow(ax,5.4,4.5,5.8,2.9,RED)
    box(ax,8.3,1.5,2,1.3,'UNKNOWN\nĐối soát receipt',GOLD);arrow(ax,9.6,4.5,9.3,2.9,GOLD)
    box(ax,10.9,1.5,1.9,1.3,'FAILED\nLỗi biết rõ',RED);arrow(ax,9.9,4.5,11.6,2.9,RED)
    arrow(ax,10,2.9,11.9,4.5,GREEN)
    ax.text(9.2,.6,'Không có receipt → vẫn UNKNOWN',ha='center',fontsize=11,color=GOLD)
    ax.text(6.5,.05,'WAITING_REVIEW và NEEDS_HUMAN là kết quả có nghĩa; không được tự gộp thành task ghi thành công.',ha='center',fontsize=11)
    save(fig,'04-state-machine',out)

    fig,ax=canvas('05 · Mất response không làm effect biến mất',
                  'Một Backend/RLock trong RAM · chưa có durability hoặc distributed exactly-once.')
    box(ax,.4,4.3,3,1.6,'CONTROLLER\nGiữ cùng intent/key K\nGửi append_note',BLUE)
    box(ax,5,4.3,3,1.6,'BACKEND COMMIT\nNote + receipt(K)\nCùng vùng lock',GREEN)
    box(ax,9.6,4.3,3,1.6,'RESPONSE BỊ MẤT\nCaller không có xác nhận\nState = UNKNOWN',RED)
    arrow(ax,3.5,5.1,4.9,5.1);arrow(ax,8.1,5.1,9.5,5.1,RED)
    box(ax,4.4,1.3,4.2,1.6,'ĐỐI SOÁT CÙNG K\nReceipt đúng → SUCCEEDED\nKhông có receipt → giữ UNKNOWN',GOLD)
    arrow(ax,10.9,4.2,8.1,3,GOLD);arrow(ax,6.4,3,6.4,4.2,GREEN)
    ax.text(1.9,2.1,'Không tự tạo key mới\nđể “thử lại”',ha='center',color=RED)
    ax.text(6.5,.35,'Cùng K + cùng action → receipt cũ, không thêm note. Cùng K + khác args → conflict.',ha='center',fontsize=12)
    save(fig,'05-idempotency',out)

    rows=r['evaluation']['rows'];states=['SUCCEEDED','FAILED','NEEDS_HUMAN','WAITING_REVIEW','CANCELLED','UNKNOWN']
    before=Counter(x['before_reconcile']['state'] for x in rows);after=Counter(x['after_reconcile']['state'] for x in rows)
    fig,ax=plt.subplots(figsize=(13,7));fig.subplots_adjust(left=.08,right=.97,bottom=.32,top=.72)
    x=np.arange(len(states));ax.bar(x-.18,[before[s] for s in states],.36,color=GOLD,label='Trước đối soát')
    ax.bar(x+.18,[after[s] for s in states],.36,color=GREEN,label='Sau đối soát')
    for i,s in enumerate(states):
        ax.text(i-.18,before[s]+.15,str(before[s]),ha='center');ax.text(i+.18,after[s]+.15,str(after[s]),ha='center')
    ax.set_xticks(x,states,rotation=15,ha='right');ax.set_ylim(0,12);ax.set_ylabel('Số case (22 Backend độc lập)');ax.legend(frameon=False)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('06 · Đạt 22/22 kỳ vọng ≠ hoàn thành 22 lần ghi',x=.055,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.055,.85,'Report scripted faults: 6 write effect; không ghi cross-tenant; 1 case vẫn UNKNOWN sau đối soát.',fontsize=11,color=GRAY)
    fig.text(.055,.11,'FAILED/CANCELLED đúng kỳ vọng vẫn không phải SUCCEEDED. Chưa đo tool choice hoặc note quality của LLM.',fontsize=11,color=RED)
    save(fig,'06-evaluation',out)
    print(json.dumps(dict(figures=6,formats=['png','svg'],cases=r['evaluation']['cases'],passed=r['evaluation']['passed'])))

if __name__=='__main__':main()
