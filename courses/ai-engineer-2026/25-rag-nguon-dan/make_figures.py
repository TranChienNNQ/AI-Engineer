"""Reproduce seven original figures from equations, annotations and an actual report."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from data import dataset
from rag import digest

OUT=Path(__file__).parent/'figures'
BLUE='#2563eb';GREEN='#0f766e';ORANGE='#c2410c';RED='#b91c1c';GRAY='#475569';INK='#172033';LIGHT='#f1f5f9'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def canvas(title,subtitle):
    fig=plt.figure(figsize=(13,7.5))
    fig.suptitle(title,x=.055,y=.963,ha='left',fontsize=20,fontweight='bold',color=INK)
    fig.text(.055,.902,subtitle,fontsize=11.5,color=GRAY)
    return fig


def diagram(fig,rect=(.045,.10,.91,.74)):
    ax=fig.add_axes(rect);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');return ax


def box(ax,x,y,w,h,label,color=BLUE,size=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.018',
                              facecolor=LIGHT,edgecolor=color,lw=1.7))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=INK,linespacing=1.65)


def arrow(ax,start,end,color=GRAY,curve=0):
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color=color,lw=1.8,connectionstyle=f'arc3,rad={curve}'))


def save(fig,name):
    OUT.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=160)
    plt.close(fig)


def pipeline():
    fig=canvas('01 · RAG: truy xuất, trả lời và giữ nguồn kiểm tra','Lab chạy offline với baseline trích xuất; không có lời gọi LLM hoặc benchmark model sinh.')
    ax=diagram(fig)
    box(ax,.02,.72,.23,.20,'Corpus + snapshot\nhiệu lực + quyền')
    box(ax,.38,.72,.23,.20,'Lọc → index\nBM25',GREEN)
    box(ax,.74,.72,.23,.20,'Query → top-6\nứng viên',GREEN)
    arrow(ax,(.25,.82),(.36,.82));arrow(ax,(.61,.82),(.72,.82))
    box(ax,.74,.31,.23,.21,'Pack budget\nE1, E2… + prompt',BLUE)
    box(ax,.38,.31,.23,.21,'Generator\nclaim + citation',BLUE)
    box(ax,.02,.31,.23,.21,'Kiểm cơ học\n→ render nguồn',ORANGE)
    arrow(ax,(.85,.71),(.85,.54));arrow(ax,(.73,.42),(.63,.42));arrow(ax,(.37,.42),(.27,.42))
    ax.text(.02,.08,'Evaluation kiểm riêng: đủ bằng chứng · claim được hỗ trợ · task hoàn thành.\nID và quote hợp lệ chưa chứng minh nội dung claim đúng.',fontsize=12,color=GRAY,linespacing=1.8)
    save(fig,'01-rag-pipeline')


def source_spans(facts):
    quote=facts['key_72']['text'];length=len(quote)
    fig=canvas('02 · Một citation phải nối lại đúng snapshot và span','Ví dụ policy tự tạo: quote đầu của P01@2; end là exclusive, offset tính theo ký tự Python/NFC.')
    ax=diagram(fig)
    box(ax,.02,.67,.34,.24,'Source P01@2\ntext + content hash\nspan nguồn: [0, '+str(length)+')',GREEN,11)
    box(ax,.63,.67,.34,.24,'Chunk C-…\nsource_id, version, start/end\ntext slice gốc',BLUE,11)
    arrow(ax,(.36,.79),(.61,.79))
    box(ax,.63,.26,.34,.23,'Evidence E1 của request\nCitation [0, '+str(length)+') tương đối\nquote khớp nguyên văn',BLUE,11)
    arrow(ax,(.80,.65),(.80,.51))
    box(ax,.02,.26,.40,.23,textwrap.fill(quote,width=43),ORANGE,11)
    arrow(ax,(.61,.37),(.44,.37))
    ax.text(.02,.05,'Nếu chunk bắt đầu ở 200 và citation [10,40): span nguồn là [210,240).\nHeader file hiển thị không được tính vào offset body.',fontsize=12,color=GRAY,linespacing=1.8)
    save(fig,'02-source-spans')


def graph(hand):
    fig=canvas('03 · Ba claim, bốn edge citation','Ví dụ tính tay: mọi ID/quote đều đúng về cơ học; quan hệ hỗ trợ nội dung vẫn có thể sai.')
    ax=diagram(fig,(.045,.22,.91,.62))
    ys=[.73,.42,.11]
    for y,text in zip(ys,['C1: key giữ 72 giờ','C2: không retry thiếu quyền','C3: mọi lỗi retry vô hạn']):
        box(ax,.02,y,.34,.16,text,BLUE,11)
    for y,text in zip(ys,['E1: policy thời hạn key','E2: retry thiếu quyền','E3: TTL cache']):
        box(ax,.65,y,.32,.16,text,GREEN,11)
    arrow(ax,(.37,.81),(.63,.81),GREEN)
    arrow(ax,(.37,.50),(.63,.50),GREEN)
    arrow(ax,(.37,.47),(.63,.19),ORANGE)
    arrow(ax,(.37,.19),(.63,.78),RED,curve=-.15)
    fig.text(.08,.105,'Attachment = 3/3       Grounded claim rate = 2/3       Citation precision = 2/4',fontsize=13,color=INK)
    fig.text(.08,.055,'Xanh: hỗ trợ claim. Cam/đỏ: không hỗ trợ. Đủ hai ý bắt buộc vẫn task fail vì C3 không có căn cứ.',fontsize=11,color=GRAY)
    save(fig,'03-claim-citation')


def versions():
    fig=canvas('04 · Phiên bản hiệu lực khác xử lý nguồn mâu thuẫn','Policy giả định P01 chuyển bản vào 2026-09-01; P01@2 chưa đặt ngày hết hiệu lực.')
    ax=fig.add_axes([.13,.43,.80,.34])
    convert=lambda s:mdates.date2num(datetime.fromisoformat(s))
    starts=[convert('2026-01-01'),convert('2026-09-01')]
    ends=[convert('2026-09-01'),convert('2027-01-01')]
    for i,(a,b,color) in enumerate(zip(starts,ends,[BLUE,GREEN])):
        ax.barh(1-i,b-a,left=a,height=.4,color=color)
    ax.set_yticks([1,0],['P01@1 · 24 giờ','P01@2 · 72 giờ'])
    ax.text(convert('2026-12-01'),0,'tiếp tục →',ha='center',va='center',color='white',fontsize=10)
    ax.set(xlim=(convert('2026-01-01'),convert('2027-01-01')),ylim=(-.5,1.5))
    for d,label,color in [('2026-06-01','Hỏi lịch sử',BLUE),('2026-11-04','Mốc serving',ORANGE)]:
        ax.axvline(convert(d),color=color,ls='--');ax.text(convert(d),1.35,label,ha='center',fontsize=10,color=color)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.grid(axis='x',alpha=.15)
    ax2=diagram(fig,(.10,.12,.83,.22))
    box(ax2,.02,.22,.39,.63,'P07: upload 20 MB\napproved, đang hiệu lực',BLUE,11)
    box(ax2,.56,.22,.39,.63,'P08: upload 25 MB\napproved, đang hiệu lực',ORANGE,11)
    ax2.text(.49,.50,'≠',ha='center',va='center',fontsize=24,color=RED)
    fig.text(.10,.045,'Chưa có authority policy để hòa giải: báo conflict có nguồn; không chọn tùy ý con số đứng đầu.',fontsize=11,color=GRAY)
    save(fig,'04-versions')


def budget(hand):
    fig=canvas('05 · Context packing phải tính cả phần ngoài body','Ví dụ tính tay, không phải token usage của LLM: 512 − 96 − 24 − 64 − 96 = 232 token cho evidence.')
    ax=fig.add_axes([.10,.47,.83,.30]);vals=[96,24,64,96,232];labels=['Instructions\n96','Query\n24','History\n64','Output\n96','Evidence\n232']
    left=0
    for value,label,color in zip(vals,labels,[GRAY,BLUE,'#64748b',ORANGE,GREEN]):
        ax.barh(0,value,left=left,height=.5,color=color)
        ax.text(left+value/2,0,label,ha='center',va='center',color='white',fontsize=10)
        left+=value
    ax.set(xlim=(0,512),ylim=(-.5,.5),xlabel='Tổng window giả định');ax.set_yticks([])
    ax2=diagram(fig,(.08,.13,.87,.22))
    box(ax2,.02,.15,.41,.70,'Theo rank: 150 trước\nCòn 82 → bỏ 120 và 112',BLUE,11)
    box(ax2,.57,.15,.40,.70,'Một lựa chọn khác: 120 + 112\nVừa 232; cần đánh giá ích lợi',GREEN,11)
    fig.text(.10,.045,'Greedy whole-chunk là baseline; đếm metadata/separator. Không cấp gold facts cho packing serving.',fontsize=11,color=GRAY)
    save(fig,'05-context-budget')


def evaluation(report):
    fig=canvas('06 · Tách kết quả baseline và fixture kiểm lỗi','Các số đếm lấy từ report thật; fixture là candidate lập trình sẵn, không là tần suất lỗi của một LLM.')
    primary=report['primary']['splits']['test'];fix=report['fixtures']['aggregate']
    ax=fig.add_axes([.08,.27,.35,.53]);values=[primary['cases'],primary['schema_valid'],primary['mechanically_accepted'],primary['task_successes']]
    names=['Query','Schema','Cơ học','Task']
    ax.bar(names,values,color=[GRAY,BLUE,GREEN,ORANGE],width=.6);ax.set(ylim=(0,14),ylabel='Số query test',title='Baseline extractive: 12 query')
    for i,v in enumerate(values):ax.text(i,v+.3,str(int(v)),ha='center')
    ax2=fig.add_axes([.56,.27,.38,.53]);values=[fix['cases'],fix['parsed'],fix['schema_valid'],fix['mechanically_accepted'],fix['task_successes']]
    names=['Case','Parse','Schema','Cơ học','Task']
    ax2.bar(names,values,color=[GRAY,BLUE,BLUE,GREEN,ORANGE],width=.6);ax2.set(ylim=(0,25),ylabel='Số fixture',title='Bộ kiểm lỗi: 22 candidate')
    for i,v in enumerate(values):ax2.text(i,v+.45,str(int(v)),ha='center')
    fig.text(.08,.11,'Baseline test: 22/22 claim có hỗ trợ nhưng task chỉ 10/12.\nHai lỗi còn lại: chưa báo conflict và trả lời đúng nguồn nhưng không đúng nhu cầu.',color=GRAY,linespacing=1.8)
    save(fig,'06-evaluation')


def ablation(report):
    fig=canvas('07 · Bằng chứng bị mất ở packing thì generator chưa được đọc','Cùng query/retriever/baseline, đổi cap context. Claim grounding vẫn bằng 1 ở cả ba cấu hình.')
    rows=report['packing_ablation'];x=np.arange(3);w=.35
    ax=fig.add_axes([.10,.27,.82,.52])
    a=[r['metrics']['packed_required_recall'] for r in rows];b=[r['metrics']['task_success_rate'] for r in rows]
    ax.bar(x-w/2,a,width=w,color=GREEN,label='Packed required recall (10 query có facts)')
    ax.bar(x+w/2,b,width=w,color=ORANGE,label='Task success (12 query test)')
    ax.set_xticks(x,[str(r['context_budget']) for r in rows]);ax.set(ylim=(0,1.25),ylabel='Tỷ lệ',xlabel='Cap serialized context (word-level token)')
    ax.legend(frameon=False,loc='upper center',fontsize=10)
    for i,(u,v) in enumerate(zip(a,b)):
        ax.text(i-w/2,u+.025,f'{u:.3f}',ha='center',fontsize=11)
        ax.text(i+w/2,v+.025,f'{v:.3f}',ha='center',fontsize=11)
    fig.text(.10,.10,'Cap 40 làm mất facts dù candidate đủ. Từ 110 lên 180 chưa sửa conflict/relevance của generator.\nKhông so các mẫu số như cùng một accuracy; số token này không thay tokenizer provider.',color=GRAY,linespacing=1.8)
    save(fig,'07-packing-ablation')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    report=json.loads((args.run/'report.json').read_text());sources,facts,queries=dataset()
    if report['corpus_hash']!=digest(sources) or report['queries_hash']!=digest(queries):raise ValueError('Stale report')
    pipeline();source_spans(facts);graph(report['hand']);versions();budget(report['hand']);evaluation(report);ablation(report)
    print('Wrote 7 PNG and 7 SVG figures to',OUT)

if __name__=='__main__':main()
