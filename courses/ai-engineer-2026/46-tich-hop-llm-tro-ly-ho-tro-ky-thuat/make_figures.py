"""Seven PNG/SVG pairs; outcome/ledger/manifest values come from the executed mock report."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BLUE='#2463a6';GREEN='#25826b';ORANGE='#dc7c30';RED='#bd4359';GRAY='#67717d'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':12,'figure.dpi':130,'savefig.dpi':150,'svg.hashsalt':'lesson46'})


def box(ax,x,y,w,h,text,color=BLUE,size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=color,ec=color,alpha=.12,lw=1.2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)


def arrow(ax,a,b,text=None,color=GRAY):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':color,'lw':1.4})
    if text:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.025,text,ha='center',fontsize=9,color=color,bbox={'facecolor':'white','edgecolor':'none','pad':1.5})


def canvas(title,size=(12,6.3)):
    fig,ax=plt.subplots(figsize=size,layout='constrained');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title,pad=15);return fig,ax


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).parent/'figures');a=p.parse_args();r=json.loads(a.report.read_text())['experiment'];a.out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for ext in ['png','svg']:
            meta={'Software':'lesson46'} if ext=='png' else {'Date':None,'Creator':'lesson46'}
            fig.savefig(a.out/(name+'.'+ext),bbox_inches='tight',facecolor='white',metadata=meta)
        plt.close(fig)
    fig,ax=canvas('01 · Thay generator, giữ authority và ticket gateway của bài 45')
    row=[(.02,'Identity + ACL\nretrieval có quyền',GREEN),(.28,'Evidence E1/E2\nquery + intent ceiling',BLUE),(.57,'HTTP adapter\nstructured candidate',ORANGE),(.83,'Parse + ground\ncurrent scope',GREEN)]
    for x,txt,col in row:box(ax,x,.65,.15,.2,txt,col,size=9)
    for x,y in [(.18,.26),(.44,.55),(.73,.81)]:arrow(ax,(x,.75),(y,.75))
    box(ax,.57,.23,.18,.19,'Draft title/body\nkhông target/quyền',ORANGE);box(ax,.26,.23,.19,.19,'Proposal → reviewer\nexact payload hash',BLUE);box(ax,.02,.23,.18,.19,'Gateway 45\nversion + receipt',GREEN)
    arrow(ax,(.9,.63),(.75,.4),'intent ghi');arrow(ax,(.55,.32),(.47,.32));arrow(ax,(.24,.32),(.22,.32))
    ax.text(.5,.53,'Mặc định: MockTransport lập trình sẵn · nhánh live riêng, chưa gọi thật',ha='center',fontsize=11)
    ax.text(.5,.08,'Model không có credential nội bộ/DB handle; instruction trong nguồn không tự cấp quyền execute',ha='center',fontsize=10)
    save(fig,'01-llm-boundary')

    fig,ax=canvas('02 · Ba câu hỏi kiểm tra khác nhau')
    for x,title,detail,col in [(.03,'Schema','Keys / types / enums\nextra-forbid / nullable',BLUE),(.36,'Citation','Evidence ID / unique quote\nserver offsets / current ACL',GREEN),(.69,'Support','Claim có được nguồn hỗ trợ?\nChưa có semantic verifier',ORANGE)]:box(ax,x,.65,.27,.21,title+'\n\n'+detail,col,size=10)
    arrow(ax,(.31,.75),(.34,.75));arrow(ax,(.64,.75),(.67,.75))
    table=ax.table(cellText=[['Supported paraphrase','Đúng','Đúng','Bị chặn vì khác string'],['Unsupported claim','Đúng','Đúng','Có thể lọt citation_only']],colLabels=['Ứng viên fixture','Schema','Citation','Policy consequence'],cellLoc='center',colWidths=[.29,.16,.16,.37],bbox=[.02,.23,.96,.28]);table.auto_set_font_size(False);table.set_fontsize(9)
    ax.text(.5,.08,'exact_quote kiểm text = quote; không chứng minh source truth, relevance hoặc completeness',ha='center',fontsize=10)
    save(fig,'02-validation-layers')

    fig,ax=canvas('03 · Retry có giới hạn và usage có phần chưa biết')
    box(ax,.04,.63,.22,.2,'Attempt 1: HTTP 503\nusage UNKNOWN',RED);box(ax,.39,.63,.22,.2,'Backoff fixture\n50 ms (fake clock)',ORANGE);box(ax,.74,.63,.22,.2,'Attempt 2: HTTP 200\n120 input / 40 output',GREEN)
    arrow(ax,(.28,.73),(.37,.73));arrow(ax,(.63,.73),(.72,.73))
    box(ax,.04,.19,.4,.23,'Có thể retry:\ntransport / 429 / 500,502,503,504\nmax 2 attempts; còn budget mới thử',BLUE,size=11)
    box(ax,.56,.19,.4,.23,'Không repair/retry tự động:\nrefusal / incomplete / schema/grounding\nHTTP 400 / unexpected tool output',ORANGE,size=11)
    ax.text(.5,.06,'Known usage ≠ total bill; timeout không có nghĩa zero cost; cooperative timeout chưa là hard API deadline',ha='center',fontsize=10)
    save(fig,'03-provider-recovery')

    fig,ax=canvas('04 · Draft là dữ liệu để duyệt, không phải action authority')
    box(ax,.04,.67,.25,.2,'Request đã kiểm\noperation / ticket / version',GREEN);box(ax,.68,.67,.27,.2,'Model candidate\nclaims + title/body draft',ORANGE)
    box(ax,.37,.37,.26,.19,'Proposal ràng buộc\nvalidated content + release\n→ reviewer xem payload/hash',BLUE)
    arrow(ax,(.24,.64),(.39,.58));arrow(ax,(.75,.64),(.61,.58))
    box(ax,.37,.05,.26,.16,'Execute gateway 45\ncommit ticket + receipt',GREEN);arrow(ax,(.5,.35),(.5,.23),'approve còn hợp lệ')
    ax.text(.03,.32,'Model không được thêm\ntenant / role / approved\nhoặc đổi target.',fontsize=10)
    ax.text(.72,.23,'UI lấy succeeded từ\nreceipt thật, không từ\ncâu model tự nhận đã làm.',fontsize=10)
    save(fig,'04-draft-to-receipt')

    pnames=['exact_quote','citation_only'];comp=r['policy_comparison'];fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    bars=axes[0].bar(pnames,[comp[p]['accuracy'] for p in pnames],color=[BLUE,ORANGE]);axes[0].bar_label(bars,labels=['11/12','11/12'],padding=3);axes[0].set(ylim=(0,1.18),ylabel='Decision accuracy',title='05 · Cùng overall accuracy')
    x=[0,1];w=.34
    axes[1].bar([v-w/2 for v in x],[comp[p]['unsafe_served'] for p in pnames],w,color=RED,label='Không nên serve nhưng lọt');axes[1].bar([v+w/2 for v in x],[comp[p]['unnecessarily_rejected'] for p in pnames],w,color=GRAY,label='Nên serve nhưng bị chặn');axes[1].set(xticks=x,xticklabels=pnames,ylim=(0,1.5),ylabel='Số case',title='Failure mode khác nhau',yticks=[0,1]);axes[1].legend(fontsize=8,loc='upper center')
    axes[2].bar([v-w/2 for v in x],[comp[p]['precision'] for p in pnames],w,color=GREEN,label='Serve precision');axes[2].bar([v+w/2 for v in x],[comp[p]['recall'] for p in pnames],w,color=BLUE,label='Serve recall');axes[2].set(xticks=x,xticklabels=pnames,ylim=(0,1.22),title='Trade-off của policy',ylabel='Tỷ lệ');axes[2].legend(fontsize=8)
    fig.suptitle('12 response cases lập trình sẵn; không phải accuracy hoặc risk rate của một LLM thật',fontsize=11)
    save(fig,'05-policy-evaluation')

    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained');axes[0].axis('off');m=r['manifest']
    text='Release '+r['release']+'\n\nRequested model: '+m['model_adapter']['model']+'\nSupport policy: '+m['support_policy']+'\nPrompt SHA: '+m['contracts']['prompt_hash'][:16]+'…\nSchema SHA: '+m['contracts']['schema_hash'][:16]+'…\nImplementation SHA: '+m['implementation_hash'][:16]+'…\nBase45 SHA: '+m['base45_source'][:16]+'…\n\nActual returned model/response ID:\nmetadata từng attempt khi có.'
    axes[0].text(.01,.95,text,va='top',fontsize=10,linespacing=1.6);axes[0].set_title('06 · Phiên bản hành vi và provenance')
    q=comp['exact_quote'];bars=axes[1].bar(['Known input','Known output'],[q['known_input_tokens'],q['known_output_tokens']],color=[BLUE,GREEN]);axes[1].bar_label(bars,padding=3);axes[1].set(ylim=(0,1650),ylabel='Token counts được đặt trong fixture',title='Một policy: 12 attempts / 1 unknown usage')
    axes[1].text(.5,.9,'1.320 / 440 chỉ là known counts\nKhông phải total bill hoặc measured usage',ha='center',transform=axes[1].transAxes,fontsize=9)
    save(fig,'06-release-usage')

    fig,ax=canvas('07 · Các mức bằng chứng khi tích hợp provider',(12,6.7))
    rows=[(.73,'Fixture + ASGI','Đã chạy: adapter / schema / scope\n78 kiểm tra mới + 75 regression bài45',GREEN),(.43,'Live smoke synthetic','Chưa chạy: account/model/schema API thật\nusage / returned model / completed/refusal',GRAY),(.13,'Domain eval + staging','Chưa chạy: semantic quality / repeats\ncost/latency / human review / remote faults',GRAY)]
    for y,title,detail,col in rows:
        box(ax,.04,y,.25,.18,title,col,size=11);box(ax,.4,y,.55,.18,detail,col,size=10);arrow(ax,(.31,y+.09),(.38,y+.09))
    arrow(ax,(.16,.71),(.16,.63));arrow(ax,(.16,.41),(.16,.33))
    ax.text(.5,.025,'Offline pass không suy model quality/production readiness; mode live phải được chạy chủ động',ha='center',fontsize=10)
    save(fig,'07-evidence-levels')
    print(json.dumps(dict(png=7,svg=7,out=str(a.out))))


if __name__=='__main__':main()
