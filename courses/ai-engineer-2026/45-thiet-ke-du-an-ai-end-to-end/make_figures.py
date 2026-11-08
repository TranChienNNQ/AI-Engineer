"""Seven static teaching diagrams; metric/manifest/audit values come from the executed report."""
import argparse,json,textwrap
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BLUE='#2463a6';GREEN='#25826b';ORANGE='#dc7c30';RED='#bd4359';GRAY='#67717d'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':12,'figure.dpi':130,'savefig.dpi':150,'svg.hashsalt':'lesson45'})


def box(ax,x,y,w,h,text,color=BLUE,size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=color,ec=color,alpha=.12,lw=1.3))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)


def arrow(ax,a,b,text=None,color=GRAY):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':color,'lw':1.5})
    if text:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.025,text,ha='center',fontsize=9,color=color,bbox={'facecolor':'white','edgecolor':'none','pad':1.5})


def canvas(title,size=(12,6.5)):
    fig,ax=plt.subplots(figsize=size,layout='constrained');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title,pad=15);return fig,ax


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).parent/'figures');a=p.parse_args();r=json.loads(a.report.read_text())['experiment'];a.out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for ext in ['png','svg']:
            meta={'Software':'lesson45'} if ext=='png' else {'Date':None,'Creator':'lesson45'}
            fig.savefig(a.out/(name+'.'+ext),bbox_inches='tight',facecolor='white',metadata=meta)
        plt.close(fig)
    fig,ax=canvas('01 · Trợ lý hỗ trợ kỹ thuật: nội dung và quyền đi qua các đường khác nhau')
    box(ax,.02,.72,.18,.15,'Client\nintent + query',BLUE);box(ax,.28,.72,.2,.15,'API contract\nidentity server',GREEN);box(ax,.58,.72,.2,.15,'Workflow\nrelease + budget',BLUE)
    arrow(ax,(.21,.795),(.26,.795));arrow(ax,(.49,.795),(.56,.795))
    box(ax,.02,.33,.2,.16,'Runbook corpus\nversion / ACL',BLUE);box(ax,.3,.33,.2,.16,'Eligible retrieval\nevidence → quote',BLUE);box(ax,.6,.33,.22,.16,'Proposal payload\nreviewer riêng',ORANGE)
    arrow(ax,(.23,.41),(.28,.41));arrow(ax,(.68,.71),(.68,.51),'intent ghi');arrow(ax,(.6,.71),(.5,.51),'nhánh đọc')
    box(ax,.62,.06,.2,.13,'Execute gateway\nquyền / TTL / version',RED,size=10);box(ax,.03,.06,.25,.13,'SQLite local\nticket + receipt + audit',GREEN)
    arrow(ax,(.71,.32),(.71,.205));arrow(ax,(.6,.125),(.3,.125),'transaction')
    ax.text(.87,.58,'Model adapter\n(nhánh mở rộng)\n\nKhông có quyền\nDB trực tiếp',ha='center',fontsize=10,color=GRAY)
    ax.text(.04,.25,'Source text là dữ liệu; không được tự chọn tool hoặc cấp quyền ghi',fontsize=10)
    save(fig,'01-system-architecture')

    fig,ax=canvas('02 · Lineage: source được phép → evidence đã pack → output → proposal')
    labels=[('Corpus','6 record / có bản cũ\nkhác tenant / chưa approved',BLUE),('Eligibility','Query timeout 504\nagent A / gateway → v2',GREEN),('Evidence','Whole paragraph\n190 chars / budget 600',BLUE),('Output','Exact quote + offsets\nversion + content hash',BLUE),('Proposal','Operation / target / args\ncitation + release → hash',ORANGE)]
    for i,(title,txt,color) in enumerate(labels):
        x=.015+i*.2;box(ax,x,.48,.17,.26,title+'\n\n'+txt,color,size=9)
        if i<4:arrow(ax,(x+.175,.61),(x+.195,.61))
    ax.text(.5,.31,'Gateway v1 / tenant B / poison chưa duyệt không vào context',ha='center',fontsize=12,color=RED)
    ax.text(.5,.16,'Citation khớp nguồn không tự chứng minh câu trả lời đủ ý hoặc sự cố đã được sửa',ha='center',fontsize=11)
    save(fig,'02-evidence-lineage')

    fig,ax=canvas('03 · State machine: proposal là ý định đã lưu, chưa là ticket effect')
    box(ax,.04,.63,.2,.19,'PENDING\nconcrete payload/hash',BLUE);box(ax,.4,.63,.2,.19,'APPROVED\nreviewer + expiry',ORANGE);box(ax,.77,.63,.19,.19,'EXECUTED\nreceipt đã commit',GREEN)
    arrow(ax,(.25,.73),(.38,.73),'review cùng tenant');arrow(ax,(.61,.73),(.75,.73),'execute đúng scope')
    box(ax,.32,.15,.38,.22,'Chặn effect mới nếu:\nexpiry / version cũ / revoke\npayload đổi / release đổi',RED,size=12)
    arrow(ax,(.5,.61),(.5,.39),color=RED)
    ax.text(.5,.04,'Các lỗi này không tự đổi state thành “executed”; cần xử lý nguyên nhân / proposal mới',ha='center',fontsize=10)
    ax.text(.02,.44,'Execute request chỉ có key:\nkhông được thay target/args\nsau khi review',fontsize=10)
    ax.text(.76,.43,'Retry cùng key:\nđọc receipt cũ,\nkhông effect lần hai',fontsize=10)
    save(fig,'03-write-state-machine')

    fig,ax=canvas('04 · Một transaction local và hai kiểu lỗi có kết quả khác nhau')
    stages=[('Validate\napproval/version',BLUE),('Mutate ticket\n+ receipt/audit',ORANGE),('COMMIT',GREEN),('HTTP response',BLUE)]
    for i,(txt,col) in enumerate(stages):
        x=.04+i*.25;box(ax,x,.64,.17,.2,txt,col,size=11)
        if i<3:arrow(ax,(x+.18,.74),(x+.235,.74))
    box(ax,.05,.17,.35,.24,'Lỗi trước COMMIT\n→ rollback ticket / receipt / state\n→ chưa có effect đã commit',RED,size=11)
    box(ax,.58,.17,.35,.24,'Lỗi sau COMMIT\n→ caller chưa biết kết quả\n→ retry key đọc receipt bền vững',GREEN,size=11)
    arrow(ax,(.37,.62),(.24,.43),'before_commit',RED);arrow(ax,(.72,.62),(.75,.43),'lost response',RED)
    ax.text(.5,.04,'Boundary chỉ gồm SQLite; remote ticket service cần idempotency/status/reconciliation riêng',ha='center',fontsize=10)
    save(fig,'04-commit-recovery')

    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    trials=r['trials'];bars=axes[0].bar([f"min_terms={v['min_terms']}" for v in trials],[v['dev_exact'] for v in trials],color=[GRAY,BLUE]);axes[0].bar_label(bars,fmt='%.3f',padding=3);axes[0].set(ylim=(0,1.2),ylabel='Exact source / abstain match',title='05 · Chọn bằng 6 development cases')
    vals=[r['always_abstain_exact'],r['test']['exact'],r['test']['answer_rate'],r['test']['answerable_recall']];bars=axes[1].bar(['Abstain\nexact','Selected\nexact','Answer\nrate','Answerable\nrecall'],vals,color=[GRAY,BLUE,ORANGE,GREEN]);axes[1].bar_label(bars,fmt='%.2f',padding=3);axes[1].set(ylim=(0,1.2),ylabel='Tỷ lệ trên 8 test cases',title='Test có 4 answerable + 4 unanswerable')
    fig.suptitle('Fixture cùng corpus nhỏ; chưa đo semantic LLM quality hoặc năng suất thật',fontsize=11)
    save(fig,'05-evaluation-results')

    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained');axes[0].axis('off')
    text='Release '+r['release']+'\n\n'+'\n'.join(f'{k}: {v}' for k,v in r['manifest'].items() if k!='corpus_hash')+'\n\ncorpus SHA-256: '+r['manifest']['corpus_hash'][:20]+'…'
    axes[0].text(.02,.95,text,va='top',fontsize=10,linespacing=1.65);axes[0].set_title('06 · Manifest behavior tối thiểu')
    counts=Counter(x['kind'] for x in r['audit']);bars=axes[1].bar(list(counts),list(counts.values()),color=[BLUE,ORANGE,GREEN]);axes[1].bar_label(bars,padding=3);axes[1].set(ylim=(0,12),ylabel='Record trong project.sqlite của run',title='Audit: run / review / effect')
    fig.suptitle('Hash không là chữ ký; audit không raw text nhưng result/proposal/ticket vẫn lưu text',fontsize=11)
    save(fig,'06-release-observability')

    fig,ax=canvas('07 · Lộ trình tăng phạm vi theo bằng chứng, chưa phải deployment đã thực hiện',(12,7))
    stages=[('Local baseline','Đã kiểm: ASGI + SQLite\ncontracts / scopes / faults',GREEN),('Read-only shadow','Cần: dữ liệu/quyền thật\nsemantic eval / logging',GRAY),('Staging writes','Cần: adapter/status/dedup\nreview / version / timeout',GRAY),('Limited pilot','Cần: quality/cost/latency\neffort reviewer / rollback',GRAY),('Mở rộng','Cần: SLO + feedback\nmeasured value + support',GRAY)]
    for i,(title,txt,color) in enumerate(stages):
        y=.82-i*.18;box(ax,.05,y,.25,.12,title,color,size=11);box(ax,.4,y,.55,.12,txt,color,size=10)
        arrow(ax,(.315,y+.06),(.38,y+.06))
        if i<4:arrow(ax,(.175,y-.008),(.175,y-.055))
    ax.text(.5,.015,'LLM / vector retrieval / remote ticket là các mở rộng cần test mới; không tự được coi là đã hoàn thành',ha='center',fontsize=10)
    save(fig,'07-delivery-roadmap')
    print(json.dumps(dict(png=7,svg=7,out=str(a.out))))


if __name__=='__main__':main()
