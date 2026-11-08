import argparse,json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'lesson48'})
BLUE='#2563a6';GREEN='#17816b';RED='#b63e55';ORANGE='#dc8732';GRAY='#647180'
def save(fig,out,name):
    for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=150,bbox_inches='tight',facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
def box(ax,x,y,w,h,label,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor=color,edgecolor=color,alpha=.1,lw=1.5));ax.text(x+w/2,y+h/2,label,ha='center',va='center')
def arrow(ax,x,y,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.8))
def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('figures'));a=p.parse_args();r=json.loads(a.report.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True);rows={x['id']:x['result'] for x in r['rows']}
    fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.03,.63,.2,.23,'Task + principal\nTrusted backend');box(ax,.3,.63,.2,.23,'Context / history\nObservation có nguồn');box(ax,.57,.63,.18,.23,'Model đề xuất\nTool hoặc terminal');box(ax,.82,.63,.15,.23,'Runtime gate\nQuyền / budget',RED)
    for start,end in [(.23,.29),(.5,.56),(.75,.81)]:arrow(ax,start,.75,end,.75)
    box(ax,.57,.18,.25,.23,'Dispatcher → tool đọc\nKết quả có call_id',GREEN);arrow(ax,.89,.61,.78,.42);arrow(ax,.57,.29,.4,.29);arrow(ax,.4,.29,.4,.61)
    ax.text(.14,.3,'Final / handoff chỉ sau kiểm\nKhông có write tool',ha='center',fontsize=12);ax.set_title('01 · Observe → propose → guard → act → update',fontsize=17,pad=18);save(fig,out,'01-agent-loop')
    fig,ax=plt.subplots(figsize=(13,4.8));ax.axis('off')
    table=ax.table(cellText=[['RAG một lượt','Code retrieval → model answer','Câu trả lời có nguồn'],['Workflow cố định','Code chọn graph theo goal','Các bước dự đoán được'],['Agent','Model chọn tool từ observation','Bước tiếp có thể đổi; host vẫn kiểm']],colLabels=['Thiết kế','Ai chọn bước tiếp?','Điều cần quan sát'],cellLoc='center',colWidths=[.2,.43,.37],loc='center');table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,3)
    ax.set_title('02 · Thêm quyền lựa chọn cho model cần thêm bằng chứng',fontsize=16,pad=20);ax.text(.5,.04,'Trong offline lab, quyết định model do AdaptiveScript lập trình; chưa LLM sampling.',transform=ax.transAxes,ha='center');save(fig,out,'02-rag-workflow-agent')
    fig,ax=plt.subplots(figsize=(13,5));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    for y,name in zip([.8,.58,.36,.14],['triage','retry_only','health_first','health_unknown']):
        rr=rows[name];tools=[o['tool'].replace('read_','').replace('search_','') for o in rr['observations']]
        ax.text(.01,y,name,va='center',fontsize=11)
        for i,label in enumerate(tools+[rr['state']]):box(ax,.22+i*.19,y-.07,.16,.14,label,GREEN if label=='answered' else ORANGE if label=='handoff' else BLUE)
    ax.set_title('03 · Đường tool và kết thúc khác theo task / observation',fontsize=16,pad=20);save(fig,out,'03-observation-paths')
    fig,ax=plt.subplots(figsize=(13,5.5));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.03,.67,.27,.22,'Response output\nreasoning item + function_call\ncall_id = call_1');box(ax,.37,.67,.26,.22,'Host validate + dispatch\nread_ticket(ticket_id=1)',RED);box(ax,.7,.67,.27,.22,'Observation O1\nfunction_call_output\ncall_id = call_1',GREEN)
    arrow(ax,.31,.78,.36,.78);arrow(ax,.64,.78,.69,.78)
    box(ax,.15,.15,.7,.28,'Input lượt tiếp: original messages → output items giữ nguyên thứ tự\n→ function_call_output đúng ID → model chọn bước mới\nOpaque reasoning chỉ phục vụ protocol, không render thành answer')
    arrow(ax,.83,.65,.75,.45);ax.set_title('04 · Tool call và tool result là hai protocol items khác nhau',fontsize=16,pad=18);save(fig,out,'04-call-correlation')
    fig,ax=plt.subplots(figsize=(13,5));ax.axis('off')
    data=[['Tên/args schema','resolve_ticket; tenant field thừa','Chặn trước dispatch'],['Task scope','Ticket 1 → đòi đọc ticket 2','Chặn đúng target'],['Correlation','Call ID đã thấy','Không dùng lại observation'],['Loop / budget','Lặp cùng tool+args; vượt giới hạn','Stop hữu hạn'],['Final gate','Quote không có / thiếu health','Không chấp nhận final']]
    t=ax.table(cellText=data,colLabels=['Lớp kiểm','Ví dụ fixture','Kết quả host'],cellLoc='center',colWidths=[.22,.48,.3],loc='center');t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2.6);ax.set_title('05 · Model đề xuất không đồng nghĩa runtime phải thực thi',fontsize=16,pad=20);save(fig,out,'05-runtime-guards')
    fig,axes=plt.subplots(1,2,figsize=(13,5));x=list(range(4));labels=['triage','retry','unknown health','no source'];w=.34
    for n,(key,color) in enumerate([('agent_reads',BLUE),('baseline_reads',ORANGE)]):axes[0].bar([v+(n-.5)*w for v in x],[c[key] for c in r['comparison']],w,label='Agent fixture' if n==0 else 'Fixed workflow',color=color)
    for n,(key,color) in enumerate([('agent_model_calls',BLUE),('baseline_model_calls',ORANGE)]):axes[1].bar([v+(n-.5)*w for v in x],[c[key] for c in r['comparison']],w,label='Agent fixture' if n==0 else 'Fixed workflow',color=color)
    for ax,title in zip(axes,['Tool read attempts (cùng kết quả)','Model calls (chưa phải tiền hoặc latency)']):ax.set_xticks(x,labels,rotation=12);ax.set_title(title);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,5.2);ax.legend(fontsize=9)
    fig.suptitle('06 · Bốn ca đơn giản: chưa chứng minh lợi ích từ agent',fontsize=16,y=1.03);save(fig,out,'06-workflow-baseline')
    fig,axes=plt.subplots(1,2,figsize=(13,5));counts=Counter(x['result']['state'] for x in r['rows']);keys=['answered','handoff','blocked','failed','stopped'];vals=[counts[k] for k in keys];bars=axes[0].bar(keys,vals,color=[GREEN,ORANGE,RED,GRAY,BLUE])
    for b,v in zip(bars,vals):axes[0].text(b.get_x()+b.get_width()/2,v+.15,str(v),ha='center')
    axes[0].set_ylim(0,max(vals)+2);axes[0].set_title(f"{r['cases']} programmed scenarios: không phải model success rate");axes[1].axis('off');axes[1].text(.03,.88,f"{r['model_calls']} model calls / {r['http_attempts']} HTTP attempts\nKnown fixture tokens: {r['known_input_tokens']} input\n{r['known_output_tokens']} output; unknown attempts: {r['unknown_usage_attempts']}\n\n27/27 khớp contract kỳ vọng\n69 kiểm tra mới + 153 hồi quy\n0 live provider calls\nKhông có ticket effect",va='top',fontsize=14)
    fig.suptitle('07 · Báo cả stop/failure và usage chưa biết',fontsize=16,y=1.03);save(fig,out,'07-outcomes-usage')
    print(json.dumps(dict(png=7,svg=7,out=str(out))))
if __name__=='__main__':main()
