"""Original scientific/teaching figures; measured panels read an actual lab report."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from data import dataset
from retrieval import chunk_sources, digest, windows

OUT=Path(__file__).parent/'figures'
BLUE='#2563eb';GREEN='#0f766e';ORANGE='#c2410c';INK='#172033';GRAY='#475569';LIGHT='#f1f5f9'
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
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=INK,linespacing=1.7)


def arrow(ax,start,end):
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.7))


def save(fig,name):
    OUT.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=160)
    plt.close(fig)


def pipeline():
    fig=canvas('01 · Chunk → embedding → ứng viên → rerank','Lab: snapshot public, neural model nhỏ học từ đầu; chưa có bước sinh câu trả lời RAG.')
    ax=diagram(fig)
    box(ax,.02,.71,.23,.22,'Nguồn + section\nversion + quyền')
    box(ax,.38,.71,.23,.22,'Chia đoạn\ntext + offset + ID',GREEN)
    box(ax,.74,.71,.23,.22,'Bi-encoder\nvector + index',GREEN)
    arrow(ax,(.25,.82),(.36,.82));arrow(ax,(.61,.82),(.72,.82))
    box(ax,.02,.30,.23,.21,'Query mới\nencode riêng',BLUE)
    box(ax,.38,.30,.23,.21,'BM25 + dense\nRRF → tối đa 6',BLUE)
    box(ax,.74,.30,.23,.21,'Query + từng chunk\ncross-encoder',ORANGE,11)
    arrow(ax,(.25,.40),(.36,.40));arrow(ax,(.61,.40),(.72,.40))
    arrow(ax,(.85,.70),(.49,.53))
    box(ax,.74,.03,.23,.16,'Top-3 có nguồn\nđo bằng chứng',ORANGE,11)
    arrow(ax,(.85,.29),(.85,.21))
    ax.text(.02,.08,'Ba lỗi: chunk đứt span · retriever bỏ sót · reranker xếp sai.\nCần đo riêng mỗi tầng để biết sửa ở đâu.',fontsize=12,color=GRAY,linespacing=1.8)
    save(fig,'01-pipeline')


def window_figure():
    fig=canvas('02 · Overlap có thể giữ bằng chứng qua ranh giới','Ví dụ tính tay: 13 token, size=6, overlap=2. Bằng chứng là token [5,9).')
    ax=fig.add_axes([.12,.25,.82,.56]);ax.set(xlim=(-.1,13.1),ylim=(-.8,5.5),xlabel='Token index (khoảng [start,end), end không được lấy)')
    rows=[('Không overlap · 1',(0,6),BLUE),('Không overlap · 2',(6,12),BLUE),('Không overlap · 3',(12,13),BLUE),
          ('Overlap · 1',(0,6),GREEN),('Overlap · 2',(4,10),GREEN),('Overlap · 3',(8,13),GREEN)]
    for y,(label,(a,b),color) in enumerate(rows[::-1]):
        ax.barh(y,b-a,left=a,height=.55,color=color,alpha=.8)
        ax.text(a+.1,y,f'[{a},{b})',color='white',va='center',fontsize=10)
    ax.axvspan(5,9,color=ORANGE,alpha=.15)
    ax.set_yticks(range(6),[r[0] for r in rows[::-1]],fontsize=10)
    ax.set_xticks(range(14));ax.grid(axis='x',alpha=.2)
    fig.text(.12,.10,'Không overlap: không chunk nào giữ trọn [5,9). Có overlap: [4,10) giữ trọn.\nChi phí body: 17 lượt token thay vì 13; bằng chứng dài hơn 6 token vẫn cần cách xử lý khác.',color=GRAY,linespacing=1.8)
    save(fig,'02-windows')


def provenance(sources,queries):
    source=sources[0];cs=[c for c in chunk_sources(sources) if c['source_id']==source['id']]
    evidence=queries[0]['evidence'];length=len(source['text'])
    fig=canvas('03 · Từ source đến chunk: giữ vị trí bằng chứng','Ví dụ thật S01/version1. Offset là ký tự trong source NFC; màu cam là canonical evidence span.')
    ax=fig.add_axes([.18,.25,.75,.55]);rows=[('Toàn source',0,length,GRAY)]
    rows += [('Section '+str(i+1),s['start'],s['end'],BLUE) for i,s in enumerate(source['sections'])]
    rows += [('Chunk '+str(i+1),c['start'],c['end'],GREEN) for i,c in enumerate(cs)]
    for y,(label,a,b,color) in enumerate(rows[::-1]):
        ax.barh(y,b-a,left=a,height=.55,color=color,alpha=.75)
        left=max(a,evidence['start']);right=min(b,evidence['end'])
        if left<right:ax.barh(y,right-left,left=left,height=.25,color=ORANGE)
    ax.set_yticks(range(len(rows)),[r[0] for r in rows[::-1]])
    ax.set(xlim=(0,length*1.02),xlabel='Ký tự Python trong source canonical');ax.grid(axis='x',alpha=.2)
    fig.text(.10,.10,'Mọi chunk giữ source_id, version, start/end và text slice gốc.\nMột chunk khác cùng source có thể không chứa câu bằng chứng, dù SourceHit vẫn bằng 1.',color=GRAY,linespacing=1.8)
    save(fig,'03-provenance')


def bi_cross():
    fig=canvas('04 · Bi-encoder và cross-encoder khác cách tương tác','Sơ đồ kiến trúc, không phải bằng chứng neural model nào luôn xếp tốt hơn.')
    ax=diagram(fig)
    box(ax,.02,.68,.19,.19,'Query → f(q)',BLUE)
    box(ax,.02,.33,.19,.19,'Chunk → g(c)',GREEN)
    box(ax,.29,.49,.19,.23,'Hai vector\ncosine / dot',BLUE)
    arrow(ax,(.21,.77),(.28,.65));arrow(ax,(.21,.42),(.28,.54))
    ax.text(.02,.10,'Document vector cache được\nđộc lập với query.\nLab: word-mean + projection.',fontsize=11,color=GRAY,linespacing=1.8)
    ax.plot([.53,.53],[.07,.94],color='#cbd5e1')
    box(ax,.60,.65,.36,.24,'[CLS] query [SEP] chunk [SEP]',ORANGE,11)
    box(ax,.60,.29,.36,.21,'Self-attention hai chiều\n→ masked mean → logit',ORANGE)
    arrow(ax,(.78,.63),(.78,.52))
    ax.text(.60,.04,'Token hai phía tương tác trong cùng forward.\nMột cặp mới cần được chấm lại.',fontsize=11,color=GRAY,linespacing=1.8)
    save(fig,'04-bi-cross')


def ceiling():
    fig=canvas('05 · Reranker chỉ đổi thứ tự trong tập ứng viên','Ví dụ giả định: chỉ B chứa trọn span; chưa phải số đo của model lab.')
    ax=diagram(fig)
    box(ax,.02,.66,.25,.24,'Ứng viên A, B, C\nB ở hạng 2\nRR@3 = 1/2',BLUE,11)
    box(ax,.39,.66,.25,.24,'Rerank B, A, C\nB lên hạng 1\nRR@3 = 1',GREEN,11)
    arrow(ax,(.27,.78),(.37,.78))
    box(ax,.02,.22,.25,.23,'Ứng viên A, C, D\nKhông có B',BLUE)
    box(ax,.39,.22,.25,.23,'Mọi hoán vị\nvẫn thiếu B',ORANGE)
    arrow(ax,(.27,.34),(.37,.34))
    box(ax,.74,.38,.23,.31,'CandidateHit = 0\n→ final Hit = 0\nDù reranker tốt',ORANGE,11)
    ax.text(.02,.04,'EvidenceHit@k sau rerank ≤ CandidateHit@M ≤ coverage của index',fontsize=13,color=INK)
    save(fig,'05-candidate-ceiling')


def evaluation(report):
    fig=canvas('06 · Coverage và thứ hạng phải được đo riêng','Số đo thật trên 18 query test answerable; no-answer báo riêng. Bộ nhỏ, chung intent giữa các split.')
    ax=fig.add_axes([.08,.29,.36,.51]);rows=report['chunk_ablation'];x=np.arange(3)
    ax.bar(x,[r['splits']['test']['coverage'] for r in rows],color=[BLUE,GRAY,GREEN],width=.65)
    ax.set_xticks(x,['Fixed24','Overlap24/8','Section48'],fontsize=10);ax.set(ylim=(0,1.13),ylabel='Coverage canonical span',title='Chunker giữ được trọn bằng chứng?')
    for i,r in enumerate(rows):ax.text(i,r['splits']['test']['coverage']+.03,f'{r["chunks"]} chunks',ha='center',fontsize=10)
    ax2=fig.add_axes([.57,.29,.37,.51]);methods=['bm25','dense','hybrid','rerank'];x=np.arange(4);w=.35
    for split,offset,color in [('dev',-w/2,BLUE),('test',w/2,ORANGE)]:
        values=[report['evaluation'][split][m]['aggregate']['rr'] for m in methods]
        ax2.bar(x+offset,values,width=w,color=color,label=split)
    ax2.set_xticks(x,methods,fontsize=10);ax2.set(ylim=(0,1.13),ylabel='MRR@3 (canonical span)',title='Thêm model chưa chắc tăng chất lượng')
    ax2.legend(frameon=False,loc='upper right',fontsize=10)
    fig.text(.08,.11,'SourceHit@3 = 1 ở cả ba chunker nhưng coverage khác nhau.\nReranker của lab thua hybrid trên dev/test; kết quả này không đánh giá model pretrained hiện đại.',color=GRAY,linespacing=1.8)
    save(fig,'06-evaluation')


def learning(report):
    fig=canvas('07 · Loss train giảm chưa đủ để chọn checkpoint','Hai loss khác nhau; chọn checkpoint theo dev MRR@3, tie giữ epoch sớm. Số liệu từ report train thật.')
    for rect,key,title in [([.08,.27,.36,.52],'bi_training','Bi-encoder: contrastive CE'),([.57,.27,.36,.52],'cross_training','Cross-encoder: weighted BCE')]:
        obj=report[key];history=obj['history'];ax=fig.add_axes(rect)
        ax.plot([r['epoch'] for r in history],[r['train_loss'] for r in history],color=BLUE,label='Train loss')
        ax.set(xlabel='Epoch',ylabel='Loss riêng của model',title=title);ax.tick_params(axis='y',colors=BLUE)
        other=ax.twinx();selected=[r for r in history if 'dev_mrr3' in r]
        other.plot([r['epoch'] for r in selected],[r['dev_mrr3'] for r in selected],color=ORANGE,marker='o',label='Dev MRR@3')
        other.set(ylim=(0,1),ylabel='Dev MRR@3');other.tick_params(axis='y',colors=ORANGE)
        ax.axvline(obj['best_epoch'],color=GREEN,ls='--',lw=1.7)
        ax.text(.05,.88,f'Chọn epoch {obj["best_epoch"]}',transform=ax.transAxes,color=GREEN,fontsize=11)
        ax.grid(alpha=.15)
    fig.text(.08,.10,'Xanh: train loss. Cam: dev MRR@3. Nét đứt: checkpoint được chọn.\nĐường dev không phải accuracy train; không dùng loss hai mô hình để so chất lượng trực tiếp.',color=GRAY,linespacing=1.8)
    save(fig,'07-learning-curves')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    report=json.loads((args.run/'report.json').read_text());sources,queries=dataset()
    if report['corpus_hash']!=digest(sources) or report['queries_hash']!=digest(queries):
        raise ValueError('Report does not match current source/query data')
    pipeline();window_figure();provenance(sources,queries);bi_cross();ceiling();evaluation(report);learning(report)
    print('Wrote 7 PNG and 7 SVG figures to',OUT)

if __name__=='__main__':main()
