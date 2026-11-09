import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BLUE='#2865a5';GREEN='#16816b';ORANGE='#df882e';RED='#b24356';GRAY='#617080'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'lesson52'})
def box(ax,x,y,w,h,text,color=BLUE,size=12):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor=color,alpha=.11,edgecolor=color,lw=1.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=1.8,color=GRAY))
def canvas(title):
 fig,ax=plt.subplots(figsize=(13,6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');fig.suptitle(title,fontsize=17,y=.98);return fig,ax

def save(fig,out,name):
 fig.savefig(out/(name+'.png'),dpi=145,bbox_inches='tight',facecolor='white');fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white',metadata={'Date':None});plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('figures'));a=p.parse_args();r=json.loads(a.report.read_text());rows={x['id']:x for x in r['rows']};out=a.out;out.mkdir(parents=True,exist_ok=True)
 fig,ax=canvas('01 · Durable state, context và cross-run memory có vai trò khác nhau')
 box(ax,.02,.60,.27,.24,'Checkpoint / state\nplan · raw observations\npending · budget',BLUE)
 box(ax,.02,.12,.27,.24,'Memory store\nsource · review · scope\nTTL · conflict · forget',GREEN)
 box(ax,.40,.36,.24,.26,'Context packet\nview ở một lượt\ntrong budget',ORANGE)
 box(ax,.77,.36,.21,.26,'Scripted planner\nđề xuất action\nchưa là authority',BLUE,size=11)
 arrow(ax,.30,.71,.39,.55);arrow(ax,.30,.24,.39,.43);arrow(ax,.65,.49,.76,.49)
 ax.text(.68,.15,'Host giữ current permission, plan và final evidence gates.',ha='center',fontsize=11,color=RED)
 save(fig,out,'01-state-context-memory')
 fig,ax=canvas('02 · Ghi memory có nguồn; candidate chưa được recall')
 stages=[('Source registry\ncontroller tin cậy',BLUE),('Validate + stage\ncandidate',ORANGE),('Reviewer approve\nactive',GREEN),('Recall hiện hành\nđúng scope / nguồn',GREEN)]
 for i,(text,col) in enumerate(stages):
  x=.025+i*.25;box(ax,x,.58,.20,.23,text,col,size=11)
  if i<3:arrow(ax,x+.205,.69,x+.24,.69)
 box(ax,.17,.15,.66,.23,'Update / revoke / TTL / conflict / forget\nCó thể làm record không còn được đưa vào context',RED)
 ax.text(.5,.03,'Review trong lab là fixture call; không human study hoặc general poison classifier.',ha='center',fontsize=11)
 save(fig,out,'02-memory-lifecycle')
 compact=rows['warm_compact']['after']['contexts'];full=rows['warm_full']['after']['contexts'];last=compact[-1];mandatory=last['mandatory_bytes'];optional=last['input_bytes']-mandatory;unused=last['available_bytes']-last['input_bytes'];reserve=2000
 fig,ax=plt.subplots(figsize=(13,5));left=0
 for label,val,col in [('Mandatory',mandatory,BLUE),('Memory thêm',optional,GREEN),('Input còn trống',unused,'#c9d3dd'),('Output reserve',reserve,ORANGE)]:
  ax.barh([0],[val],left=left,color=col,height=.45,label=f'{label}: {val} B');ax.text(left+val/2,0,str(val),ha='center',va='center',fontsize=11);left+=val
 ax.set_xlim(0,14000);ax.set_ylim(-.65,.65);ax.set_yticks([]);ax.set_xlabel('Byte JSON UTF-8 theo teaching budget, không token');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2);ax.spines[['top','right','left']].set_visible(False)
 fig.suptitle('03 · Ví dụ packet cuối warm_compact: giữ mandatory trước, thêm memory nếu vừa',fontsize=16,y=.98);save(fig,out,'03-context-budget')
 fig,axes=plt.subplots(1,2,figsize=(13,5.3));turns=[1,2,3,4]
 for label,data,col in [('compact',compact,BLUE),('full',full,ORANGE)]:axes[0].plot(turns,[c['input_bytes'] for c in data],'-o',label=label,color=col)
 axes[0].set_xticks(turns);axes[0].set_xlabel('Planning turn');axes[0].set_ylabel('Serialized input bytes');axes[0].legend();axes[0].set_title('Lượt 1: compact lớn hơn 3 B')
 totals=[sum(c['input_bytes'] for c in compact),sum(c['input_bytes'] for c in full)];bars=axes[1].bar(['compact','full'],totals,color=[BLUE,ORANGE])
 for b,v in zip(bars,totals):axes[1].text(b.get_x()+b.get_width()/2,v+300,str(v),ha='center')
 axes[1].set_ylim(0,31000);axes[1].set_title(f'Tổng: chênh {totals[1]-totals[0]} B, cùng actions/final')
 for ax in axes:ax.spines[['top','right']].set_visible(False)
 fig.suptitle('04 · So byte packet trên fixture; chưa đo token / bill / LLM quality',fontsize=16,y=1.02);fig.tight_layout();save(fig,out,'04-context-bytes')
 fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');fig.suptitle('05 · Lọc scope / lifecycle / source trước khi ranking và packing',fontsize=16,y=.98)
 table=ax.table(cellText=[['Sai tenant/owner/service','Không vào candidate pool'],['Candidate / superseded / forgotten','Không recall'],['expires_at ≤ now','Expired'],['Source revoked / version/hash cũ','Không dùng nguồn dẫn xuất cũ'],['Hint trigger không khớp','Irrelevant'],['Cùng key, khác payload','Conflict: loại cả nhóm, cần resolve'],['Các item còn lại','Priority / recency / byte budget']],colLabels=['Điều kiện','Kết quả của lab'],cellLoc='center',colWidths=[.48,.52],loc='center');table.auto_set_font_size(False);table.set_fontsize(12);table.scale(1,2.6)
 for (i,j),cell in table.get_celld().items():
  if i==0:cell.set_facecolor('#e8eef5')
 save(fig,out,'05-memory-gates')
 cross=r['cross_run'];aa=cross['run_a']['state'];bb=cross['run_b']['state'];fig,ax=canvas('06 · Run B dùng hint để chọn query, vẫn đọc source / health hiện hành')
 box(ax,.02,.56,.27,.26,f"Run A: cold\n{aa['turns']} planning / {aa['attempts']} attempts\nsearch rỗng → đổi query",BLUE)
 box(ax,.38,.56,.24,.26,'Source → candidate\ncontroller review\nactive hint',ORANGE,size=11)
 box(ax,.71,.56,.27,.26,f"Run B: warm\n{bb['turns']} planning / {bb['attempts']} attempts\nquery phù hợp từ đầu",GREEN)
 arrow(ax,.30,.69,.37,.69);arrow(ax,.63,.69,.70,.69)
 box(ax,.14,.13,.72,.24,'Hai run IDs / checkpoint chains riêng · một memory store\nMemory không thay runbook citation hoặc current health',GREEN)
 ax.text(.5,.02,'Policy scripted và mapping synthetic; chưa bằng chứng lợi ích LLM trên population.',ha='center',fontsize=11)
 save(fig,out,'06-cross-run')
 fig,ax=canvas('07 · Forget phải được kiểm cả ở packet mới và checkpoint cũ')
 box(ax,.02,.56,.27,.25,'Forget logical key\nclear memory payload/refs\ntombstone + epoch tăng',RED,size=11)
 box(ax,.39,.56,.24,.25,'Packet đã compose?\ncheck lại trước consume\nmemory_snapshot_changed',ORANGE,size=10)
 box(ax,.73,.56,.25,.25,'Resume checkpoint cũ?\nbinding mismatch\nkhông phục hồi hint cũ',ORANGE,size=10)
 arrow(ax,.30,.69,.38,.69);ax.plot([.16,.16,.855],[.55,.46,.46],color=GRAY,lw=1.7);arrow(ax,.855,.46,.855,.55)
 box(ax,.15,.11,.70,.24,'Raw source / checkpoint query / backup vẫn có thể còn\nLogical forget chưa là semantic hoặc physical erasure',BLUE,size=12)
 save(fig,out,'07-forget-resume')
 print(json.dumps(dict(png=7,svg=7,out=str(out))))
if __name__=='__main__':main()
