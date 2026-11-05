"""Six original figures from contract rules and measured deterministic fixtures."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
ROOT=Path(__file__).resolve().parent
BLUE,TEAL,ORANGE,GRAY='#2563eb','#0f766e','#c2410c','#475569'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','figure.facecolor':'white'})

def save(fig,name):
    for ext in ('png','svg'):fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
def box(ax,x,y,text,w=.25,h=.18,color=BLUE):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.01',ec=color,fc=color+'12'))
    ax.text(x,y,text,ha='center',va='center',fontsize=11)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':GRAY,'lw':1.5})

def architecture():
    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    for x,text in [(.15,'Client nghiệp vụ\nTicket + lines'),(.5,'API ứng dụng\nInput / orchestration'),(.85,'Adapter provider\nRaw content + status')]:box(ax,x,.72,text)
    arrow(ax,(.29,.72),(.36,.72));arrow(ax,(.64,.72),(.71,.72))
    box(ax,.5,.29,'Validator\nJSON → schema → ID',w=.27,color=TEAL)
    box(ax,.85,.29,'Success / error\nKhông tự thực thi action',w=.27,color=TEAL)
    arrow(ax,(.85,.61),(.5,.40));arrow(ax,(.65,.29),(.70,.29))
    ax.set_title('Client contract tách khỏi provider contract',weight='bold',fontsize=16)
    ax.text(.5,.035,'Lab dùng provider rule/scripted cục bộ; không có LLM call hoặc side effect.',ha='center',color=GRAY)
    save(fig,'01-architecture')

def validation():
    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    texts=['1. Finish bình thường?','2. JSON parse được?','3. Đúng schema / quan hệ field?','4. Evidence ID có trong input?','5. Đúng task / được dữ kiện hỗ trợ?']
    ys=[.86,.69,.52,.35,.18]
    for i,(y,text) in enumerate(zip(ys,texts)):
        box(ax,.45,y,text,w=.68,h=.11,color=ORANGE if i==4 else BLUE)
        if i:arrow(ax,(.45,ys[i-1]-.065),(.45,y+.065))
    ax.text(.83,.18,'Scorer/reference\nhoặc kiểm semantic\nkhông tự có từ schema',ha='left',va='center',fontsize=10,color=ORANGE)
    ax.set_title('Một gate pass không tự bảo đảm gate tiếp theo',weight='bold',fontsize=16)
    save(fig,'02-validation')

def retries(report):
    rows={r['id']:r for r in report['rows']};fig,axes=plt.subplots(2,1,figsize=(12,6))
    examples=[('timeout-twice',[(400,'Attempt 1 timeout',BLUE),(50,'Backoff',GRAY),(400,'Attempt 2 timeout',ORANGE)]),
              ('retry-recovers',[(20,'Rate limit',ORANGE),(100,'Retry-after wait',GRAY),(30,'Reply OK',TEAL)])]
    for ax,(name,parts) in zip(axes,examples):
        start=0
        for length,label,color in parts:
            ax.barh([0],[length],left=start,color=color,label=label)
            ax.text(start+length/2,0,str(length),ha='center',va='center',color='white',fontsize=10,weight='bold');start+=length
        ax.set(yticks=[],ylim=(-.6,1.1),xlabel='ms (FakeClock, không phải benchmark)',xlim=(0,start*1.12),title=f"{name}: {rows[name]['attempts']} attempts, {rows[name]['simulated_elapsed_ms']:g} ms")
        ax.legend(loc='upper left',ncol=3,fontsize=9)
    fig.suptitle('Tổng deadline không reset giữa các attempt',weight='bold',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.93),h_pad=3);save(fig,'03-retry')

def funnel(report):
    keys=['complete','json_valid','schema_valid','accepted','correct'];s=report['summary'];n=s['samples']
    counts=[round(s[k]*n) for k in keys]
    fig,axes=plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios':[1.2,1]})
    labels=['Completion','JSON sau completion','Schema sau các gate','Accepted (gồm abstain)','Task success']
    bars=axes[0].barh(labels,counts,color=[GRAY,BLUE,BLUE,TEAL,ORANGE]);axes[0].invert_yaxis()
    axes[0].bar_label(bars,labels=[f'{c}/{n} = {100*c/n:g}%' for c in counts],padding=4,fontsize=10)
    axes[0].set(xlim=(0,25),xlabel='Số case',title='Giữ mẫu số N=20')
    axes[1].axis('off');axes[1].text(.02,.88,'Accepted accuracy: 5/8 = 62.5%\n\nEnd-to-end success: 5/20 = 25%\n\nRoute coverage: 7/20 = 35%\n\nRoute accuracy: 4/7 ≈ 57.14%\n\nKhông tráo mẫu số để làm đẹp metric.',va='top',fontsize=13)
    fig.suptitle('20 fixture chủ đích: không là ước lượng chất lượng một LLM thật',weight='bold',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'04-eval-funnel')

def matrices(report):
    fig,axes=plt.subplots(1,2,figsize=(13,6));c=report['confusion'];m=np.array(c['counts'])
    axes[0].imshow(m,cmap='Blues',vmin=0,vmax=max(m.max(),1),aspect='auto')
    for (i,j),v in np.ndenumerate(m):axes[0].text(j,i,str(v),ha='center',va='center',color='white' if v>6 else 'black')
    axes[0].set_xticks(range(5),c['columns'],rotation=30);axes[0].set_yticks(range(4),c['rows']);axes[0].set(xlabel='Kết quả pipeline',ylabel='Expected',title='Giữ ERROR khác ABSTAIN')
    slices=report['slices'];names=list(slices);x=np.arange(len(names))
    axes[1].barh(x,[slices[n]['n'] for n in names],color='#cbd5e1',label='Tổng case')
    axes[1].barh(x,[slices[n]['correct'] for n in names],color=TEAL,label='Đúng task')
    axes[1].set_yticks(x,names);axes[1].invert_yaxis();axes[1].set(xlabel='Số case',title='Slice clean không đại diện toàn suite');axes[1].legend(fontsize=9)
    fig.suptitle('Lỗi format, transport và semantic cần được nhìn riêng',weight='bold',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.93));save(fig,'05-confusion-slices')

def plan():
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,title,lines,color in [(axes[0],'Đã chạy trong bài 22',['Fixture có output/lỗi định trước','Parser / schema / evidence ID','ASGI HTTP status + OpenAPI','Retry với FakeClock','Report tái lập, hash fixture'],BLUE),
                                  (axes[1],'Khi nối model thật',['Dataset theo task, reference review','Tách dev/validation/holdout theo group','Version prompt/model/scorer + settings','Metric task/coverage/error/latency/cost','Kiểm slice + so baseline cùng case'],TEAL)]:
        ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title(title,weight='bold')
        ys=[.87,.70,.53,.36,.19]
        for i,(y,line) in enumerate(zip(ys,lines)):
            box(ax,.5,y,line,w=.94,h=.10,color=color)
            if i:arrow(ax,(.5,ys[i-1]-.06),(.5,y+.06))
    fig.suptitle('Contract/fault test và model quality eval bổ sung cho nhau',weight='bold',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'06-eval-plan')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text())['evaluation']
    architecture();validation();retries(report);funnel(report);matrices(report);plan()
    print('Saved 6 PNG + 6 SVG')
if __name__=='__main__':main()
