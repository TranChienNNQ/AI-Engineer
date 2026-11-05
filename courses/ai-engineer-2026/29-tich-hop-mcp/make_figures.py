"""Six reproducible MCP teaching figures: real report counters, architecture diagrams, no LLM benchmark."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

ROOT=Path(__file__).resolve().parent
BLUE='#2863a5';GREEN='#227b62';RED='#bd4949';GRAY='#596777';GOLD='#d89a24'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})


def canvas(title,subtitle,size=(13,7.5)):
    fig,ax=plt.subplots(figsize=size);ax.set(xlim=(0,13),ylim=(0,7));ax.axis('off')
    fig.suptitle(title,x=.055,y=.96,ha='left',fontsize=19,fontweight='bold',color='#172d45')
    fig.text(.055,.89,subtitle,fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.04,right=.98,bottom=.08,top=.84)
    return fig,ax


def box(ax,x,y,w,h,label,color=BLUE,fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07,rounding_size=0.12',
                              facecolor=color+'12',edgecolor=color,linewidth=1.5))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color='#172d45',linespacing=1.6)


def arrow(ax,x,y,xx,yy,color=GRAY):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=15,color=color,lw=1.5))


def save(fig,name,out):
    for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=170,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'figures');args=p.parse_args()
    if args.report:r=json.loads(args.report.read_text())
    else:
        import asyncio
        from lab import report
        r=asyncio.run(report())
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    fig,ax=canvas('01 · Host giữ agent; MCP nối tới backend',
                  'Host/client/server là vai trò phần mềm. Server không bắt buộc có model.')
    box(ax,.4,2.4,3.2,3.2,'AI HOST\nTask · model / policy\nQuyền · context\nState / memory\nMCP client manager',BLUE)
    box(ax,4.6,4.3,2.6,1.3,'MCP CLIENT A\nProtocol + transport',GOLD)
    box(ax,4.6,1.8,2.6,1.3,'MCP CLIENT B\nProtocol + transport',GOLD)
    box(ax,8.3,4.3,2.2,1.3,'MCP SERVER A\nTools / resources',GREEN)
    box(ax,8.3,1.8,2.2,1.3,'MCP SERVER B\nTools / prompts',GREEN)
    box(ax,11.4,4.3,1.2,1.3,'Ticket\nbackend',BLUE,10)
    box(ax,11.4,1.8,1.2,1.3,'KB\nbackend',BLUE,10)
    arrow(ax,3.7,4.7,4.5,4.9);arrow(ax,3.7,3,4.5,2.5)
    arrow(ax,7.3,4.9,8.2,4.9);arrow(ax,7.3,2.5,8.2,2.5)
    arrow(ax,10.6,4.9,11.3,4.9);arrow(ax,10.6,2.5,11.3,2.5)
    ax.text(6.5,.6,'MCP chuẩn hóa giao tiếp; task policy và kiểm quyền backend vẫn cần được thiết kế.',ha='center',fontsize=12)
    save(fig,'01-host-client-server',out)

    fig,ax=canvas('02 · Ba primitive: tool, resource, prompt',
                  'List để khám phá, call/read/get để sử dụng. Host quyết định cách đưa kết quả vào task.')
    entries=[(4.7,'TOOLS','get_ticket\nsearch_runbooks','tools/list → tools/call',BLUE),
             (2.8,'RESOURCES','kb://catalog\nkb://runbooks/{doc_id}','resources/list hoặc\nresources/templates/list\n→ resources/read URI cụ thể',GREEN),
             (.9,'PROMPTS','triage_ticket(ticket_id)','prompts/list → prompts/get\nRender template, chưa gọi tool',GOLD)]
    for y,title,example,method,color in entries:
        box(ax,.4,y,2.1,1.4,title,color);box(ax,3.4,y,3.4,1.4,example,color)
        box(ax,7.7,y,4.8,1.4,method,color);arrow(ax,2.6,y+.7,3.3,y+.7);arrow(ax,6.9,y+.7,7.6,y+.7)
    ax.text(6.5,.25,'Resource body và prompt template không tự có thẩm quyền sửa system instruction.',ha='center',fontsize=11)
    save(fig,'02-primitives',out)

    fig,ax=canvas('03 · Revision hiện hành khác đời handshake',
                  'Đối chiếu 05/11/2026 · SDK lab 2.3.0 · protocol modern 2026-07-28, forced legacy 2025-11-25.')
    ax.text(.4,6.1,'2026-07-28',fontsize=13,fontweight='bold',color=GREEN)
    box(ax,.4,4.3,3.2,1.4,'server/discover\n1 request ↔ 1 response',GREEN)
    box(ax,4.8,4.3,3.2,1.4,'tools/list\ninput/output schema',BLUE)
    box(ax,9.2,4.3,3.2,1.4,'tools/call\nresultType + isError',BLUE)
    arrow(ax,3.7,5,4.7,5);arrow(ax,8.1,5,9.1,5)
    ax.text(.4,3.4,'Handshake era',fontsize=13,fontweight='bold',color=GOLD)
    box(ax,.4,1.8,3.2,1.2,'initialize\nRequest / response',GOLD)
    box(ax,4.8,1.8,3.2,1.2,'notifications/initialized\nKhông có response',GOLD,10)
    box(ax,9.2,1.8,3.2,1.2,'Bắt đầu list / call\nSau handshake',BLUE)
    arrow(ax,3.7,2.4,4.7,2.4);arrow(ax,8.1,2.4,9.1,2.4)
    ax.text(6.5,.6,'Modern: từng request mang _meta version/capabilities; discovery không là authentication.',ha='center',fontsize=11)
    save(fig,'03-protocol-flow',out)

    fig,ax=canvas('04 · Cùng API, ba đường kiểm có phạm vi khác nhau',
                  'Cả ba dùng SDK thật; chỉ stdio tạo child process trong bộ kiểm tự động.')
    entries=[(4.7,'IN-MEMORY','Client(server object)','Protocol dispatch trong process\nKhông pipe, không HTTP/TCP',BLUE),
             (2.8,'STDIO','Client → child server.py','stdin/stdout: JSON-RPC mỗi dòng\nLog qua stderr; không cần port',GREEN),
             (.9,'HTTP-ASGI','Client → HTTP endpoint /mcp','HTTP request/ASGI trong process\nChưa socket, TLS, proxy hoặc OAuth',GOLD)]
    for y,title,client,label,color in entries:
        box(ax,.4,y,2.2,1.4,title,color);box(ax,3.5,y,3.4,1.4,client,color,10)
        box(ax,7.8,y,4.7,1.4,label,color);arrow(ax,2.7,y+.7,3.4,y+.7);arrow(ax,7,y+.7,7.7,y+.7)
    ax.text(6.5,.2,'Server HTTP local có entrypoint riêng để tự thử; bài chưa benchmark/deploy qua mạng.',ha='center',fontsize=11)
    save(fig,'04-transports',out)

    fig,ax=canvas('05 · Giao thức chuẩn chưa thay policy và kiểm dữ liệu',
                  'Workspace là cấu hình deployment trong lab; chưa là authentication nhiều người dùng.')
    box(ax,.4,4.4,3.2,1.7,'HOST POLICY\nAlias + allowlist + task\nSchema / budget / timeout\nOutput / snapshot hash',GREEN,10)
    box(ax,4.9,4.4,3.2,1.7,'MCP EXCHANGE\nDiscovery + list + call\nResult hoặc protocol error',BLUE)
    box(ax,9.4,4.4,3.2,1.7,'BACKEND GUARD\nWorkspace tại lúc đọc\nResolve URI / trả dữ liệu',GREEN)
    arrow(ax,3.7,5.3,4.8,5.3);arrow(ax,8.2,5.3,9.3,5.3)
    box(ax,.4,1.3,3.2,1.7,'HOST / LOCAL ERROR\noutside_task\ntimeout tại client',RED)
    box(ax,4.9,1.3,3.2,1.7,'JSON-RPC ERROR\n-32601: unknown method\n-32022: version không hỗ trợ',RED,10)
    box(ax,9.4,1.3,3.2,1.7,'TOOL ERROR RESULT\nisError = true\nKhông dùng như data thành công',RED,10)
    ax.text(6.5,.35,'readOnlyHint / serverInfo / [SYSTEM] trong body không tự cấp quyền hay bảo đảm text đúng.',ha='center',fontsize=11)
    save(fig,'05-trust-and-errors',out)

    groups={
      'integration':'Transport / legacy','compatibility':'Transport / legacy',
      'host_before_dispatch':'Host policy / route','host_catalog':'Host policy / route','host_budget':'Host policy / route','host_routing':'Host policy / route',
      'server_tool':'Data / output / timeout','server_data':'Data / output / timeout','host_data_authority':'Data / output / timeout','host_output':'Data / output / timeout','host_output_or_local':'Data / output / timeout',
      'protocol':'Protocol / SDK behavior','sdk_observed_behavior':'Protocol / SDK behavior','host_parser_unit':'Parser / schema unit'}
    counts=Counter(groups[x['layer']] for x in r['evaluation']['rows'])
    labels=list(counts);vals=[counts[x] for x in labels]
    wire=r['wire']['transcript'];rpc=sum('error' in x['response'] for x in wire)
    tools=sum(bool(x['response'].get('result',{}).get('isError')) for x in wire)
    success=len(wire)-rpc-tools
    fig,axes=plt.subplots(1,2,figsize=(14,7));fig.subplots_adjust(left=.2,right=.97,bottom=.28,top=.73,wspace=.7)
    axes[0].barh(np.arange(len(labels)),vals,color=BLUE);axes[0].set_yticks(np.arange(len(labels)),labels,fontsize=10);axes[0].invert_yaxis();axes[0].set_xlim(0,15)
    axes[0].set_title('33 expected-outcome checks');axes[0].set_xlabel('Số kiểm tra')
    for i,v in enumerate(vals):axes[0].text(v+.3,i,str(v),va='center')
    axes[1].bar(['Result OK','Tool error','RPC error'],[success,tools,rpc],color=[GREEN,GOLD,RED],width=.65)
    axes[1].set_title('13 request của wire observer');axes[1].set_ylim(0,11);axes[1].set_ylabel('Số response')
    for i,v in enumerate([success,tools,rpc]):axes[1].text(i,v+.2,str(v),ha='center',fontsize=12)
    for ax in axes:ax.grid(axis='x' if ax is axes[0] else 'y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('06 · Đếm theo tầng, giữ riêng residual và khác biệt SDK',x=.055,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.055,.86,'33/33 predicate đạt; không là conformance toàn MCP hoặc task accuracy của LLM.',fontsize=11,color=GRAY)
    fig.text(.055,.12,'Residual answer-only BANANA_OVERRIDE chưa bị chặn; không có semantic verifier trong lab.',fontsize=12,color=RED)
    fig.text(.055,.05,'5 logical data operations của workflow là một mẫu số khác; chưa đo tổng RPC/cost/latency production.',fontsize=11,color=GRAY)
    save(fig,'06-evaluation',out)
    print(json.dumps(dict(figures=6,formats=['png','svg'],cases=r['evaluation']['cases'],protocol=r['metadata']['protocol'])))

if __name__=='__main__':main()
