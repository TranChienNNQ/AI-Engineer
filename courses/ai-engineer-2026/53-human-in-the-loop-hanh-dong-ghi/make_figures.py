import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BLUE='#2865a5';GREEN='#16816b';ORANGE='#df882e';RED='#b24356';GRAY='#617080'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'lesson53'})
def box(ax,x,y,w,h,text,color=BLUE,size=12):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor=color,alpha=.11,edgecolor=color,lw=1.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=1.8,color=GRAY))
def canvas(title):
 fig,ax=plt.subplots(figsize=(13,6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');fig.suptitle(title,fontsize=17,y=.98);return fig,ax

def save(fig,out,name):
 fig.savefig(out/(name+'.png'),dpi=145,bbox_inches='tight',facecolor='white');fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white',metadata={'Date':None});plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('figures'));a=p.parse_args();r=json.loads(a.report.read_text());rows={x['id']:x for x in r['rows']};out=a.out;out.mkdir(parents=True,exist_ok=True)
 fig,ax=canvas('01 · Proposal, approval, effect và receipt là các mốc khác nhau')
 stages=[('Investigation\ncó nguồn',BLUE),('Draft\nchưa cấp quyền',BLUE),('Reviewer\nduyệt đúng bản',ORANGE),('Target write\ncó thể unknown',ORANGE),('Receipt/status\nconfirmed',GREEN)]
 for i,(text,col) in enumerate(stages):
  x=.02+i*.20;box(ax,x,.57,.16,.23,text,col,size=11)
  if i<4:arrow(ax,x+.165,.68,x+.195,.68)
 box(ax,.14,.14,.72,.22,'Approved chưa là đã gửi · Đã gửi chưa là đã ghi\nConfirmed note chưa là sự cố đã được khắc phục',GREEN)
 ax.text(.5,.02,'Review decisions và draft trong lab là fixture; chưa LLM hoặc human study.',ha='center',fontsize=11)
 save(fig,out,'01-review-write-boundaries')
 fig,ax=canvas('02 · Reviewer xem một view cụ thể; approval bind đúng draft/version/expiry')
 box(ax,.025,.54,.28,.28,'Đối tượng / tác động\nticket 1 · expected v1\nappend INTERNAL note\nstatus giữ open',BLUE,size=11)
 box(ax,.36,.54,.28,.28,'Nội dung / nguồn\nbody đầy đủ + citations\nroot cause chưa rõ\nchưa remediation',BLUE,size=11)
 box(ax,.695,.54,.28,.28,'Identity / thời hạn\ndraft hash + version\nworkflow revision\napprove expiry 160',ORANGE,size=11)
 arrow(ax,.315,.68,.35,.68);arrow(ax,.65,.68,.685,.68)
 box(ax,.17,.14,.66,.22,'Edit → hash/version mới → approval cũ bị hủy\nLấy view mới và review lại trước execute',RED)
 save(fig,out,'02-review-payload')
 fig,ax=canvas('03 · Local transaction không atomically bao trùm target effect')
 box(ax,.02,.54,.27,.27,'Local approval DB\ncommit in_flight\nreserve attempt',BLUE)
 box(ax,.37,.54,.27,.27,'Target DB transaction\nnote + version + receipt\ncommit cùng nhau',GREEN,size=11)
 box(ax,.72,.54,.26,.27,'Local result DB\nverify receipt\ncommit confirmed',BLUE)
 arrow(ax,.30,.68,.36,.68);arrow(ax,.65,.68,.71,.68)
 ax.text(.33,.43,'POST',ha='center',fontsize=11);ax.text(.68,.43,'ACK có thể mất',ha='center',fontsize=11,color=RED)
 box(ax,.14,.10,.72,.23,'Crash trước POST: reservation có nhưng chưa có effect\nCrash sau target commit: local chưa biết dù note đã ghi',ORANGE)
 save(fig,out,'03-transaction-gap')
 fig,ax=canvas('04 · Unknown outcome: tra status, giữ key/payload, không tự đoán no effect')
 box(ax,.02,.57,.23,.24,'Unknown\nkhông có local receipt\nchưa biết effect',ORANGE,size=11)
 box(ax,.38,.70,.25,.17,'Status applied\nreceipt identity khớp',GREEN,size=11)
 box(ax,.38,.31,.25,.20,'not_found / unavailable\nvẫn unknown',ORANGE,size=11)
 box(ax,.76,.70,.22,.17,'Confirmed\nkhông POST mới',GREEN,size=11)
 box(ax,.76,.24,.22,.27,'Explicit same-key retry\ncurrent approval/scope\n+ remaining budget',BLUE,size=10)
 arrow(ax,.26,.70,.37,.78);arrow(ax,.26,.59,.37,.42);arrow(ax,.64,.78,.75,.78);arrow(ax,.64,.41,.75,.38)
 ax.text(.5,.07,'Fixture status_missing_retry: 2 POST handlers · 1 note effect; target dedup trong transaction.',ha='center',fontsize=11)
 save(fig,out,'04-unknown-reconcile')
 selected=['approved','status_missing_retry','before_send_retry','before_commit_retry','crash_before_send','attempt_budget'];labels=['normal','lost status','before send','target rollback','crash before send','budget stop'];fig,ax=plt.subplots(figsize=(13,5.4));x=range(len(selected));w=.24
 for i,(label,col) in enumerate([('attempt reserved',BLUE),('POST handlers',ORANGE),('note effects',GREEN)]):
  vals=[]
  for name in selected:
   rr=rows[name]['after'];vals.append(rr['local']['state']['attempts'] if i==0 else sum(q['method']=='POST' for q in rr['target']['requests']) if i==1 else len(rr['target']['notes']))
  ax.bar([k+(i-1)*w for k in x],vals,w,label=label,color=col)
 ax.set_xticks(list(x),labels,rotation=15,ha='right');ax.set_ylim(0,2.7);ax.set_yticks([0,1,2]);ax.legend(ncol=3);ax.spines[['top','right']].set_visible(False);fig.suptitle('05 · Reservation, HTTP handler và effect có mẫu số khác nhau',fontsize=16,y=1.02);fig.tight_layout();save(fig,out,'05-attempts-effects')
 fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');fig.suptitle('06 · Chặn fresh write không xóa một effect đã xảy ra',fontsize=16,y=.98)
 table=ax.table(cellText=[['Owner active / đúng tenant','Bắt buộc','Bắt buộc'],['Approval hiện còn hạn','Bắt buộc','Không cấp write mới'],['Source/memory current','Bắt buộc trước send','Không export source/note body'],['Target version CAS','Cho effect chưa có','Receipt là version lịch sử'],['Operation/draft identity','Bind permit + payload','Bind receipt + status'],['Kết quả','Có thể gửi một write attempt','Chỉ xác nhận effect cũ']],colLabels=['Điều kiện / mục đích','Fresh POST','Reconcile operation cũ'],cellLoc='center',colWidths=[.34,.31,.35],loc='center');table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,2.8)
 for (i,j),cell in table.get_celld().items():
  if i==0:cell.set_facecolor('#e7eef5')
 save(fig,out,'06-current-rights-history')
 fig,ax=canvas('07 · Human approval là một lớp; host và target vẫn giữ các gate')
 entries=[('Draft / source gate','schema · exact sources · uncertainty',BLUE),('Review gate','reviewer scope · view hash · edit invalidation',ORANGE),('Execution gate','current identity/source · expiry · persistent budget',ORANGE),('Target transaction','permit binding · version CAS · dedup receipt',GREEN),('Result / reconciliation','op + draft + target identity · unknown không bị giấu',GREEN)]
 for i,(title,sub,col) in enumerate(entries):
  y=.82-i*.16;box(ax,.12,y-.09,.76,.12,title+'\n'+sub,col,size=10)
  if i<4:arrow(ax,.5,y-.105,.5,y-.135)
 ax.text(.5,.01,'Public demo HMAC key chưa là production security; confirmed note chưa là remediation.',ha='center',fontsize=10,color=RED)
 save(fig,out,'07-control-layers')
 print(json.dumps(dict(png=7,svg=7,out=str(out))))
if __name__=='__main__':main()
