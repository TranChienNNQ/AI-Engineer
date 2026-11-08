import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BLUE='#2563a6';GREEN='#16806d';ORANGE='#df872b';RED='#b14256';GRAY='#617080'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':15,'svg.hashsalt':'lesson51'})
def box(ax,x,y,w,h,text,color=BLUE,size=12):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',facecolor=color,alpha=.11,edgecolor=color,linewidth=1.7));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=1.8,color=GRAY))
def canvas(title):
 fig,ax=plt.subplots(figsize=(13,6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');fig.suptitle(title,fontsize=17,y=.98);return fig,ax

def save(fig,out,name):
 fig.savefig(out/(name+'.png'),dpi=145,bbox_inches='tight',facecolor='white');fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white',metadata={'Date':None});plt.close(fig)
def table(title,labels,rows,widths):
 fig,ax=plt.subplots(figsize=(13,5.8));ax.axis('off');fig.suptitle(title,fontsize=16,y=.98)
 t=ax.table(cellText=rows,colLabels=labels,cellLoc='center',loc='center',colWidths=widths);t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2.7)
 for (i,j),cell in t.get_celld().items():
  if i==0:cell.set_facecolor('#e7eef5')
 return fig

def main():
 p=argparse.ArgumentParser();p.add_argument('--report',required=True,type=Path);p.add_argument('--out',type=Path,default=Path('figures'));a=p.parse_args();r=json.loads(a.report.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True);rows={x['id']:x for x in r['rows']}
 fig,ax=canvas('01 · State của công việc tiếp tục sống sau worker')
 box(ax,.03,.65,.23,.18,'ready\nchọn bước',BLUE)
 box(ax,.385,.65,.23,.18,'planning\nđã dành lượt',ORANGE)
 box(ax,.74,.65,.23,.18,'pending\naction đã lưu',ORANGE)
 arrow(ax,.275,.74,.37,.74);arrow(ax,.63,.74,.725,.74)
 box(ax,.385,.22,.23,.18,'final\nvalidate + commit',GREEN)
 box(ax,.74,.22,.23,.18,'Observation commit\nxóa pending',GREEN,size=11)
 arrow(ax,.50,.635,.50,.415);arrow(ax,.855,.635,.855,.415)
 ax.text(.51,.52,'terminal',fontsize=11)
 ax.plot([.855,.855,.145],[.205,.12,.12],color=GRAY,lw=1.6);arrow(ax,.145,.12,.145,.635)
 ax.text(.16,.43,'→ ready',fontsize=11)
 ax.text(.5,.035,'Pause sau observation commit; lỗi / budget có thể kết thúc run ở boundary khác.',ha='center',fontsize=11)
 save(fig,out,'01-durable-state')
 fig,ax=canvas('02 · Commit ngắn trước / sau I/O; không một transaction bao trùm API ngoài')
 items=[('Reserve\nplanning turn',BLUE),('Guard + commit\nplan / intent',BLUE),('Reserve\nattempt',BLUE),('MCP call\nngoài transaction',ORANGE),('Commit observation\nxóa pending',GREEN)]
 for i,(text,color) in enumerate(items):
  x=.02+i*.20;box(ax,x,.56,.16,.23,text,color,size=11)
  if i<4:arrow(ax,x+.165,.67,x+.195,.67)
 box(ax,.13,.13,.74,.23,'Mỗi checkpoint + head update: một SQLite transaction\nCrash trong transaction → bản chưa commit không được công bố',GREEN)
 ax.text(.5,.44,'Khoảng trước/sau external call cần recovery policy riêng.',ha='center',color=RED)
 save(fig,out,'02-commit-protocol')
 fig=table('03 · Cùng “chưa có observation” nhưng thế giới bên ngoài có thể khác',
 ['Crash boundary','Counter / kết quả trên đĩa','Recovery của lab'],[
 ['Sau intent','Pending, chưa reserve attempt','Dành attempt rồi đọc'],
 ['Sau reserve, trước dispatch','Counter đã tăng, chưa gọi','Không hoàn budget; retry đọc'],
 ['Sau response, trước commit','Tool đã trả; checkpoint chưa biết','Unknown gap → retry đọc hữu hạn'],
 ['Sau observation commit','Kết quả đã lưu atomically','Replay, không gọi lại operation'],
 ['Giữa terminal transaction','INSERT chưa commit/head chưa đổi','Rollback; proposal mới tốn lượt'],
 ['Sau final commit','Final đã lưu, chưa deliver','Revalidate rồi trả lại final']], [.27,.39,.34])
 save(fig,out,'03-crash-windows')
 row=rows['after_response'];b=row['before'];e=row['after'];fig,ax=canvas('04 · after_response: kết quả tool tồn tại ngoài checkpoint trước crash')
 box(ax,.03,.53,.39,.30,f"Trước crash\npending: S2 / attempt 1\nreservations={b['state']['attempts']} · ack={len(b['acknowledgements'])} · obs={len(b['state']['observations'])}",ORANGE)
 box(ax,.58,.53,.39,.30,f"Sau worker mới\nretry S2 → S4 → health → final\nreservations={e['state']['attempts']} · ack={len(e['acknowledgements'])} · obs={len(e['state']['observations'])}",GREEN)
 arrow(ax,.44,.68,.56,.68)
 box(ax,.14,.12,.72,.22,'support-504:S2 giữ nguyên op_id; attempt tăng 1 → 2\nProbe thấy 2 acknowledgements của S2, state chỉ lưu 1 observation',BLUE,size=12)
 ax.text(.5,.02,'Probe là test instrument; runtime không dùng probe làm recovery oracle.',ha='center',fontsize=11)
 save(fig,out,'04-recovery-trace')
 names=['uninterrupted','planning_reserved','attempt_reserved','after_response','observation_committed','terminal_transaction'];labels=['normal','lost plan','before send','after response','after commit','partial final'];fig,axes=plt.subplots(1,2,figsize=(13,5.4));x=list(range(len(names)));w=.36
 axes[0].bar(x,[rows[n]['after']['state']['turns'] for n in names],color=BLUE);axes[0].set_title('Planning reservations')
 for i,(label,key,color) in enumerate([('attempt reserved','reserved',ORANGE),('probe ack','ack',GREEN)]):
  vals=[rows[n]['after']['state']['attempts'] if key=='reserved' else len(rows[n]['after']['acknowledgements']) for n in names];axes[1].bar([k+(i-.5)*w for k in x],vals,w,label=label,color=color)
 axes[1].set_title('Reservation khác acknowledged execution');axes[1].legend(fontsize=10)
 for ax in axes:
  ax.set_xticks(x,labels,rotation=22,ha='right');ax.set_ylim(0,7);ax.spines[['top','right']].set_visible(False);ax.set_yticks(range(8))
 fig.suptitle('05 · Replay bỏ qua bước đã commit; retry có thể tốn thêm attempt / I/O',fontsize=16,y=1.03);fig.tight_layout();save(fig,out,'05-reservations-and-acks')
 fig,ax=canvas('06 · Revision + epoch fence việc ghi checkpoint')
 box(ax,.02,.55,.25,.25,'Worker 1\nepoch 1 / revision 10',BLUE)
 box(ax,.375,.55,.25,.25,'Controller đợi W1 chết\nclaim epoch 2',GREEN,size=11)
 box(ax,.73,.55,.25,.25,'Worker 2 save\nkhớp epoch + head',GREEN,size=11)
 arrow(ax,.285,.68,.36,.68);arrow(ax,.64,.68,.715,.68)
 box(ax,.15,.13,.70,.23,'Old epoch → stale_owner · Old head → stale_revision\nMCP server không kiểm epoch: chưa fence external effects',RED,size=12)
 ax.text(.5,.025,'Không lease / heartbeat / leader election; không takeover worker còn sống trong suite.',ha='center',fontsize=11)
 save(fig,out,'06-ownership')
 fig,ax=canvas('07 · Resume không được bỏ qua các lớp kiểm hiện tại')
 entries=[('Authenticate / bind','actor, tenant, task',BLUE),('Journal integrity','hash chain, schema, head',BLUE),('Release / ownership','source/config + epoch/revision',ORANGE),('Current evidence','ticket, source, health, scope',ORANGE),('Budget / action / final','không reset; validate rồi export',GREEN)]
 for i,(title,sub,color) in enumerate(entries):
  y=.82-i*.16;box(ax,.17,y-.09,.66,.12,title+' · '+sub,color,size=11)
  if i<4:arrow(ax,.5,y-.105,.5,y-.135)
 ax.text(.5,.01,'Chặn export sau revoke không xóa raw checkpoint / backup.',ha='center',fontsize=11,color=RED)
 save(fig,out,'07-resume-gates')
 print(json.dumps(dict(png=7,svg=7,out=str(out))))
if __name__=='__main__':main()
