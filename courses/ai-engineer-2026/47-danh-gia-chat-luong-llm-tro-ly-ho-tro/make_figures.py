"""Static teaching figures; quantitative panels read the executed report."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'lesson47'})
BLUE='#2563a6';GREEN='#17816b';RED='#b63e55';ORANGE='#dc8732';GRAY='#647180'

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight',facecolor='white')
    fig.savefig(out/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});plt.close(fig)
def box(ax,x,y,w,h,text,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor=color,alpha=.1,edgecolor=color,lw=1.5))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',wrap=True)
def arrow(ax,x,y,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.8))
def bars(ax,labels,values,title,color=BLUE,percent=False):
    bars_=ax.bar(labels,values,color=color,width=.6)
    for b,v in zip(bars_,values):ax.text(b.get_x()+b.get_width()/2,v+.02 if percent else v+.1,f'{v:.1%}' if percent else str(v),ha='center',va='bottom')
    ax.set_title(title,pad=18);ax.spines[['top','right']].set_visible(False)
    if percent:ax.set_ylim(0,1.16)
    else:ax.set_ylim(0,max(values,default=1)*1.4+1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('figures'));a=p.parse_args();r=json.loads(a.report.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.02,.64,.2,.22,'Case + quyền\nNguồn đúng phiên bản');box(ax,.28,.64,.2,.22,'Output pipeline 46\nAnswer / draft / fail');box(ax,.54,.64,.2,.22,'Rubric + annotations\nNội dung được review');box(ax,.8,.64,.18,.22,'Metrics / slices\nDecision gate')
    for x in [.22,.48,.74]:arrow(ax,x,.75,x+.055,.75)
    box(ax,.07,.16,.25,.25,'Contract\nJSON / quote / ACL\nCó thể kiểm bằng code',GREEN);box(ax,.38,.16,.25,.25,'Semantic quality\nĐúng / đủ / liên quan\nCần bằng chứng chấm',ORANGE);box(ax,.69,.16,.25,.25,'Action quality\nDraft / scope / effect\nKhông chỉ chấm prose',RED)
    for x in [.195,.505,.815]:arrow(ax,x,.42,x,.6)
    ax.set_title('01 · Chấm chất lượng cần cả output lẫn thước đo',fontsize=16,pad=18);save(fig,out,'01-evaluation-loop')
    fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.03,.63,.4,.24,'DEV: 4 incident groups × 2 queries\n8 cases × 3 programmed repeats = 24 rows',BLUE)
    box(ax,.57,.63,.4,.24,'TEST: 4 groups khác × 2 queries\n8 cases × 3 programmed repeats = 24 rows',GREEN)
    box(ax,.03,.22,.4,.22,'Chốt rubric / configs / gate trước\nselect_dev chỉ nhận dev summaries',BLUE)
    box(ax,.57,.22,.4,.22,'Test là diagnostic đã khai báo\nKhông chọn lại winner từ test',GREEN)
    arrow(ax,.23,.62,.23,.45);arrow(ax,.77,.62,.77,.45)
    ax.text(.5,.08,'2 configs × (24 dev + 24 test) = 96 records; source/domain vẫn chung, không là unseen-document benchmark',ha='center')
    ax.set_title('02 · Query paraphrase cùng incident ở chung split',fontsize=16,pad=18);save(fig,out,'02-dataset-split')
    fig,ax=plt.subplots(figsize=(13,4.6));ax.axis('off')
    rows=[['Đủ ý + có nguồn','2','2','2','—','Không'],['Quote đúng, thiếu điều kiện retry','2','1','2','—','Không'],['Retry ghi vô hạn trái nguồn','0','0','2','—','Có'],['Answer đúng, draft bịa đã sửa','2','2','2','0','Có']]
    t=ax.table(cellText=rows,colLabels=['Ví dụ authored','Support','Đầy đủ','Liên quan','Draft','Critical'],cellLoc='center',colWidths=[.4,.12,.12,.12,.12,.12],loc='center');t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2.6)
    for (i,j),cell in t.get_celld().items():
        if i==0:cell.set_facecolor('#e8eef5')
        if j==5 and i in [3,4]:cell.set_facecolor('#f6dfe4')
    ax.set_title('03 · Điểm answer tốt không xóa lỗi critical của draft',fontsize=16,pad=14)
    ax.text(.5,.03,'Thang 0/1/2 theo rubric của lab; không phải universal grading standard',transform=ax.transAxes,ha='center');save(fig,out,'03-rubric-examples')
    cfg=list(r['summary']);labels=['quote_guard','synthesis_candidate'];colors=[BLUE,ORANGE]
    fig,axes=plt.subplots(2,2,figsize=(12,8));fig.subplots_adjust(hspace=.48,wspace=.22)
    bars(axes[0,0],labels,[r['summary'][c]['dev']['task_success'] for c in cfg],'Dev: task success / mọi rows',colors,True)
    bars(axes[0,1],labels,[r['summary'][c]['test']['task_success'] for c in cfg],'Test: task success / mọi rows',colors,True)
    bars(axes[1,0],labels,[r['summary'][c]['test']['critical'] for c in cfg],'Test: critical trong served outputs',colors)
    bars(axes[1,1],labels,[r['summary'][c]['test']['coverage'] for c in cfg],'Test: coverage = served / mọi rows',colors,True)
    fig.suptitle('04 · Scripted fixtures + authored ratings; cả hai config đều trượt dev gate',fontsize=15,y=1.01);save(fig,out,'04-policy-results')
    fig,axes=plt.subplots(1,2,figsize=(13,5));tags=['answerable','write','permission'];x=list(range(3));w=.34
    for n,c in enumerate(cfg):
        vals=[r['slices'][c]['test'][t]['task_success'] for t in tags]
        axes[0].bar([v+(n-.5)*w for v in x],vals,w,label=c,color=colors[n])
    axes[0].set_xticks(x,tags);axes[0].set_ylim(0,1.28);axes[0].legend(fontsize=9,loc='upper left');axes[0].set_title('Test: task success theo slice')
    errors=['support','completeness','draft','critical'];x=list(range(4))
    for n,c in enumerate(cfg):
        vals=[r['summary'][c]['test']['errors'].get(e,0) for e in errors]
        axes[1].bar([v+(n-.5)*w for v in x],vals,w,label=c,color=colors[n])
    axes[1].set_xticks(x,errors);axes[1].set_ylim(0,8);axes[1].set_title('Test: lỗi đa nhãn (không cộng thành N)')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('05 · Không để điểm trung bình che lỗi draft hoặc thiếu bước chẩn đoán',fontsize=15,y=1.04);save(fig,out,'05-error-slices')
    audit=r['judge_audit']['vs_author'];fig,axes=plt.subplots(1,2,figsize=(12,5))
    matrix=[[audit['tp'],audit['fn']],[audit['fp'],audit['tn']]];axes[0].imshow(matrix,cmap='Blues',vmin=0,vmax=max(map(max,matrix)))
    for i in range(2):
        for j in range(2):axes[0].text(j,i,str(matrix[i][j]),ha='center',va='center',fontsize=22,color='white' if matrix[i][j]>20 else 'black')
    axes[0].set_xticks([0,1],['Judge pass','Judge fail']);axes[0].set_yticks([0,1],['Author good','Author bad']);axes[0].set_title('Citation-present proxy trên 62 served records')
    axes[1].axis('off');axes[1].text(.02,.85,f"Agreement = {audit['agreement']:.1%}\nCohen κ = {audit['kappa']:.1f}\nBad-pass rate = {audit['bad_pass_rate']:.0%}",fontsize=15,va='top')
    axes[1].text(.02,.5,'Order 1: A / B → chọn A\nOrder 2: B / A → chọn A (vị trí)\nMap về identity: B → bất nhất\nKết quả: needs_review',fontsize=13,va='top')
    fig.suptitle('06 · Bộ chấm lập trình sẵn, không phải LLM judge hoặc nghiên cứu human agreement',fontsize=14,y=1.04);save(fig,out,'06-judge-audit')
    fig,axes=plt.subplots(1,2,figsize=(13,5));d=r['test_pair_diagnostic']
    diffs=[r['groups'][cfg[1]]['test'][g]-r['groups'][cfg[0]]['test'][g] for g in r['groups'][cfg[0]]['test']]
    axes[0].bar(['gateway','scope','upload','write'],diffs,color=BLUE);axes[0].axhline(0,color=GRAY,lw=1);axes[0].set_ylim(-.05,.4);axes[0].set_title(f"4 group deltas; mean +{d['delta']:.1%}\nBootstrap minh họa [{d['low']:.0%}, {d['high']:.0%}]")
    from scoring import wilson
    low,high=wilson(0,30);axes[1].hlines(0,low,high,color=RED,lw=6);axes[1].plot([0],[0],'o',color=RED);axes[1].set_xlim(-.01,.3);axes[1].set_ylim(-1,1);axes[1].set_yticks([]);axes[1].set_xlabel('Tỷ lệ lỗi p');axes[1].set_title(f'Ví dụ binomial riêng: 0 lỗi / 30 thử\nWilson 95% ≈ [0, {high:.1%}]')
    axes[1].text(.13,-.45,'Chỉ có ý nghĩa với giả định\nsampling phù hợp; không áp lên\n96 authored fixture rows.',ha='center',fontsize=10)
    fig.suptitle('07 · Repeats/paraphrases không tự tạo thêm incident độc lập',fontsize=15,y=1.07);save(fig,out,'07-uncertainty')
    print(json.dumps(dict(png=7,svg=7,out=str(out))))
if __name__=='__main__':main()
