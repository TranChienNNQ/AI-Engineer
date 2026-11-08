import argparse,json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'lesson49'})
BLUE='#2563a6';GREEN='#17816b';RED='#b63e55';ORANGE='#dc8732';GRAY='#647180'
def save(fig,out,name):
 for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=150,bbox_inches='tight',facecolor='white',metadata={'Date':None} if ext=='svg' else None)
 plt.close(fig)
def box(ax,x,y,w,h,text,color=BLUE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor=color,edgecolor=color,alpha=.1,lw=1.5));ax.text(x+w/2,y+h/2,text,ha='center',va='center')
def arrow(ax,x,y,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.8))
def table_figure(out,name,title,headers,rows,widths,foot=''):
 fig,ax=plt.subplots(figsize=(13,5));ax.axis('off');t=ax.table(cellText=rows,colLabels=headers,cellLoc='center',colWidths=widths,loc='center');t.auto_set_font_size(False);t.set_fontsize(10.5);t.scale(1,2.8)
 for (i,j),cell in t.get_celld().items():
  if i==0:cell.set_facecolor('#e8eef5')
 ax.set_title(title,fontsize=16,pad=20)
 if foot:ax.text(.5,.01,foot,ha='center',transform=ax.transAxes,fontsize=10)
 save(fig,out,name)
def main():
 p=argparse.ArgumentParser();p.add_argument('--report',required=True,type=Path);p.add_argument('--out',type=Path,default=Path('figures'));a=p.parse_args();r=json.loads(a.report.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True)
 fig,ax=plt.subplots(figsize=(13,5.5));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
 for x,w,text,color in [(.02,.2,'Agent48\nModel tool decision',BLUE),(.28,.2,'Host49 / Link\nCatalog + local policy',RED),(.54,.2,'MCP client/server\nMemory hoặc stdio',BLUE),(.8,.18,'World48 backend\nSynthetic SQLite',GREEN)]:box(ax,x,.59,w,.26,text,color)
 for x,y in [(.23,.27),(.49,.53),(.75,.79)]:arrow(ax,x,.72,y,.72)
 box(ax,.1,.14,.8,.25,'Model function-call protocol ≠ MCP tools/call\nHost ánh xạ logical name → server alias / remote name / wrapper args\nMCP không tự chọn kế hoạch hoặc cấp quyền cho model')
 ax.set_title('01 · Hai giao diện, một vòng agent được giữ lại',fontsize=17,pad=20);save(fig,out,'01-agent-mcp-boundary')
 table_figure(out,'02-tool-contract','02 · Tool tốt giảm phần model phải đoán',['Khía cạnh','Thiết kế dễ nhầm','Contract của lab'],[
 ['Tên / mục đích','do_action(data)','ticket_read: chỉ đọc ticket được giao'],['Input / scope','tenant, role, SQL do model chọn','ticket_id/query; task scope ở backend'],['Output / đơn vị','Một đoạn text tùy ý','Envelope version + data typed + snapshot/source'],['Lỗi / side effect','Empty = error? Retry có ghi không?','No evidence / unavailable / isError tách rõ']], [.2,.35,.45])
 fig,ax=plt.subplots(figsize=(13,5.5));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
 box(ax,.03,.62,.25,.25,'tools/list: từng page\nTên / schema / description\nAnnotations là hints');box(ax,.37,.62,.26,.25,'Host chọn đúng 3 tools\nKiểm schema / aliases\nPin catalog hash',RED);box(ax,.72,.62,.25,.25,'Model catalog curated\nGiữ contract bài48\nKhông forward mọi metadata',GREEN)
 arrow(ax,.29,.75,.36,.75);arrow(ax,.64,.75,.71,.75)
 box(ax,.15,.14,.7,.26,'Trước mỗi call: refresh full catalog → hash phải khớp\nĐổi description/schema/name → dừng, không tự chấp nhận contract mới\nHash chưa là signature; refresh/call chưa atomic')
 arrow(ax,.83,.61,.73,.42);ax.set_title('03 · Discovery, mapping và snapshot của catalog',fontsize=16,pad=20);save(fig,out,'03-catalog-pinning')
 fig,ax=plt.subplots(figsize=(13,5));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
 box(ax,.02,.6,.21,.28,'Model: read_ticket\ncall_id = call_1\nargs flat',BLUE);box(ax,.29,.6,.35,.28,'Host route\nsupport_a__ticket_read\nMCP args = {request: {...}}',RED);box(ax,.71,.6,.27,.28,'MCP tools/call\nJSON-RPC ID do SDK quản lý\nstructuredContent + isError',GREEN)
 arrow(ax,.24,.74,.28,.74);arrow(ax,.65,.74,.7,.74)
 box(ax,.12,.13,.76,.26,'Host kiểm outputSchema + local contract + đúng resource/source hash\n→ Observation O1 có mcp_provenance → function_call_output(call_1)\nKhông dùng MCP RPC ID làm model call_id hoặc ticket receipt')
 ax.set_title('04 · Liên kết model call, MCP request và observation',fontsize=16,pad=20);save(fig,out,'04-routing-and-result')
 table_figure(out,'05-error-layers','05 · Phân loại lỗi trước khi quyết định retry',['Lớp','Ví dụ','Host làm gì'],[
 ['Local policy','Sai target / args / catalog đổi','Chặn trước tools/call'],['MCP protocol','JSON-RPC error','Giữ layer/code an toàn, dừng'],['Tool result','isError=true','Observation lỗi, model có thể handoff'],['Business data','ok=false / unavailable','Giữ khác no evidence'],['Output / timeout','Sai hash/resource hoặc quá thời gian','Reject hoặc timeout; không retry MCP tự động']], [.22,.39,.39],foot='Schema đúng chưa chứng minh source thật, task đủ ý hoặc quyền hiện hành.')
 fig,axes=plt.subplots(1,2,figsize=(12,5));x=[0,1];labels=['triage','retry'];w=.32
 for n,(title,key,color) in enumerate([('Direct Agent48','model_calls_direct',BLUE),('Agent48 qua MCP','model_calls_mcp',ORANGE)]):axes[0].bar([v+(n-.5)*w for v in x],[p[key] for p in r['comparison']],w,label=title,color=color)
 for n,(title,key,color) in enumerate([('Local read attempts','reads_direct',BLUE),('MCP calls sent','mcp_calls',ORANGE)]):axes[1].bar([v+(n-.5)*w for v in x],[p[key] for p in r['comparison']],w,label=title,color=color)
 for ax,title in zip(axes,['Model calls: giữ cùng agent policy','Tool operations: cùng path / answer / citation']):ax.set_xticks(x,labels);ax.set_ylim(0,5.4);ax.set_title(title);ax.legend(fontsize=9);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('06 · MCP đổi đường tích hợp, chưa là cải tiến năng lực model',fontsize=16,y=1.03);save(fig,out,'06-direct-mcp-parity')
 counts=Counter(x['result']['state'] for x in r['rows']);fig,axes=plt.subplots(1,2,figsize=(13,5));keys=['answered','handoff','failed','blocked'];vals=[counts[k] for k in keys];bars=axes[0].bar(keys,vals,color=[GREEN,ORANGE,GRAY,RED]);axes[0].set_ylim(0,6)
 for b,v in zip(bars,vals):axes[0].text(b.get_x()+b.get_width()/2,v+.1,str(v),ha='center')
 axes[0].set_title('14 scenario: expected fail/block cũng là pass');axes[1].axis('off');axes[1].text(.02,.9,f"MCP calls sent: {r['mcp_calls_sent']}\nServer handler attempts: {r['server_attempted']}\nAdmitted backend reads: {r['server_admitted_reads']}\nModel calls fixture: {r['model_calls']}\n\n64 kiểm tra mới + 222 hồi quy\n3 scenario stdio subprocess thật\n0 provider calls / 0 ticket effects",fontsize=14,va='top')
 fig.suptitle('07 · Attempt, backend read và model turn là các mẫu số khác nhau',fontsize=16,y=1.03);save(fig,out,'07-results-and-counts')
 print(json.dumps(dict(png=7,svg=7,out=str(out))))
if __name__=='__main__':main()
