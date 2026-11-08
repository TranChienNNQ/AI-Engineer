import argparse,json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'lesson50'})
BLUE='#2563a6';GREEN='#17816b';RED='#b63e55';ORANGE='#dc8732';GRAY='#647180'
def save(fig,out,name):
 for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=150,bbox_inches='tight',facecolor='white',metadata={'Date':None} if ext=='svg' else None)
 plt.close(fig)
def box(ax,x,y,w,h,text,color=BLUE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor=color,edgecolor=color,alpha=.1,lw=1.5));ax.text(x+w/2,y+h/2,text,ha='center',va='center')
def arrow(ax,x,y,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.7))
def tablefig(out,name,title,headers,data,widths,foot=''):
 fig,ax=plt.subplots(figsize=(13,5.2));ax.axis('off');t=ax.table(cellText=data,colLabels=headers,cellLoc='center',colWidths=widths,loc='center');t.auto_set_font_size(False);t.set_fontsize(10.5);t.scale(1,2.7)
 for (i,j),cell in t.get_celld().items():
  if i==0:cell.set_facecolor('#e8eef5')
 ax.set_title(title,fontsize=16,pad=20)
 if foot:ax.text(.5,.01,foot,transform=ax.transAxes,ha='center',fontsize=10)
 save(fig,out,name)
def main():
 p=argparse.ArgumentParser();p.add_argument('--report',required=True,type=Path);p.add_argument('--out',type=Path,default=Path('figures'));a=p.parse_args();r=json.loads(a.report.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True);rows={x['id']:x['result'] for x in r['rows']}
 fig,ax=plt.subplots(figsize=(13,5.8));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
 for x,w,text,color in [(.02,.19,'Plan proposal\nBước + dependency',BLUE),(.28,.2,'Host plan guard\nDAG / scope / parent',RED),(.55,.19,'MCP execution\nObservation có nguồn',GREEN),(.8,.18,'Verifier\nPostconditions',ORANGE)]:box(ax,x,.62,w,.24,text,color)
 for x,y in [(.22,.27),(.49,.54),(.75,.79)]:arrow(ax,x,.74,y,.74)
 box(ax,.24,.13,.55,.25,'Supported → bước tiếp / answer\nEmpty / error / discrepancy → revise hoặc handoff\nKhông lấy plan text làm evidence')
 arrow(ax,.89,.6,.74,.4);arrow(ax,.24,.26,.12,.26);arrow(ax,.12,.26,.12,.6)
 ax.set_title('01 · Plan → act → observe → verify → replan',fontsize=17,pad=18);save(fig,out,'01-planning-loop')
 fig,ax=plt.subplots(figsize=(13,5.8));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
 ax.text(.01,.91,'Revision 1',fontsize=13);box(ax,.06,.64,.22,.16,'S1: read ticket',GREEN);box(ax,.39,.64,.22,.16,'S2: search query gốc\nempty ≠ satisfied',ORANGE);box(ax,.72,.64,.22,.16,'S3: read health\nchưa ready')
 arrow(ax,.29,.72,.38,.72);arrow(ax,.62,.72,.71,.72)
 ax.text(.01,.45,'Revision 2',fontsize=13);box(ax,.06,.17,.22,.16,'S1 giữ nguyên',GREEN);box(ax,.39,.17,.22,.16,'S4: query từ ticket\nsatisfied',GREEN);box(ax,.72,.17,.22,.16,'S3: phụ thuộc S4',BLUE)
 arrow(ax,.29,.25,.38,.25);arrow(ax,.62,.25,.71,.25);ax.text(.5,.04,'S2 vẫn lưu là empty; không sửa thành công và không xóa lần tìm kiếm trước.',ha='center')
 ax.set_title('02 · Đổi bước tương lai, giữ lịch sử đã thực thi',fontsize=16,pad=18);save(fig,out,'02-plan-revision-dag')
 recovery=rows['recover_memory'];data=[]
 for event in recovery['planning_trace']:
  action=event['action'];data.append([event['turn'],event['revision'],action['step_id'] or 'terminal',action['tool'] or action['kind']])
 tablefig(out,'03-recovery-trace','03 · Trace thật của fixture recover_memory',['Planning turn','Revision','Step','Lựa chọn'],data,[.2,.2,.2,.4],foot='5 lượt đề xuất / 4 MCP calls; proposal do Python script, chưa LLM sampling.')
 tablefig(out,'04-verification-states','04 · Thực thi thành công khác điều kiện đã thỏa',['Observation / dữ liệu','Check','Kết quả'],[
 ['Search ok=true, evidence=[]','runbook_available','empty'],['Search có nguồn hợp lệ','runbook_available','supported'],['Health status=unknown','health_known','unknown'],['Report degraded / snapshot ok, cùng mốc','report_consistency','discrepancy'],['Report và snapshot khác mốc','report_consistency','not_comparable'],['Chỉ có report + health + runbook','root_cause_established','unknown']], [.52,.25,.23])
 tablefig(out,'05-plan-guards','05 · Kế hoạch bị chặn trước khi biến thành tool calls',['Lỗi đề xuất','Guard','Ý nghĩa'],[
 ['Dependency tạo cycle','cyclic_plan','Không thể có bước ready hợp lệ'],['Model tự status=done','invalid_proposal_schema','Host mới quyết định trạng thái'],['Sửa query của bước đã chạy','rewrote_executed_step','Không viết lại lịch sử'],['Đổi parent hash / evidence ID','stale / unknown evidence','Revision gắn đúng dữ liệu đã thấy'],['Dùng lại trigger O2 để replan mãi','reused_revision_trigger','Đổi chữ chưa là tiến triển']], [.4,.3,.3])
 fig,axes=plt.subplots(1,2,figsize=(13,5));labels=['straight','recover search','discrepancy'];x=range(3);w=.34
 for n,(kind,color) in enumerate([('baseline',BLUE),('planned',ORANGE)]):
  axes[0].bar([i+(n-.5)*w for i in x],[p[kind]['turns'] for p in r['comparison']],w,label=kind,color=color)
  axes[1].bar([i+(n-.5)*w for i in x],[p[kind]['calls'] for p in r['comparison']],w,label=kind,color=color)
 for ax,title in zip(axes,['Lượt đề xuất theo policy fixture','MCP tool calls']):ax.set_xticks(list(x),labels,rotation=10);ax.set_ylim(0,6.5);ax.set_title(title);ax.legend(fontsize=9);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('06 · Recovery tốn thêm bước; policy discrepancy của bài 50 cũng khác baseline',fontsize=15,y=1.03);save(fig,out,'06-scripted-baseline')
 counts=Counter(x['result']['state'] for x in r['rows']);fig,axes=plt.subplots(1,2,figsize=(13,5));keys=['answered','handoff','failed','stopped'];values=[counts[k] for k in keys];bars=axes[0].bar(keys,values,color=[GREEN,ORANGE,GRAY,RED])
 for b,v in zip(bars,values):axes[0].text(b.get_x()+b.get_width()/2,v+.2,str(v),ha='center')
 axes[0].set_ylim(0,max(values)+3);axes[0].set_title('25 cases: expected reject/fail cũng là pass');axes[1].axis('off');axes[1].text(.02,.9,f"Accepted plans: {r['accepted_plans']}\nPlanning turns: {r['planning_turns']}\nMCP calls: {r['mcp_calls']}\n0 provider HTTP attempts\n\n63 kiểm tra mới + 286 hồi quy\n2 scenario stdio subprocess thật\nKhông causal proof / LLM benchmark",va='top',fontsize=14)
 fig.suptitle('07 · Đếm plans / decisions / executions riêng nhau',fontsize=16,y=1.03);save(fig,out,'07-outcomes-and-budgets')
 print(json.dumps(dict(png=7,svg=7,out=str(out))))
if __name__=='__main__':main()
