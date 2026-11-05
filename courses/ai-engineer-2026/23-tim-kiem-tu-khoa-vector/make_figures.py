"""Regenerate original PNG/SVG figures from equations and a verified lab report."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from lab import SearchIndex, normalize_rows
from data import documents

BLUE='#2563eb';GREEN='#0f766e';ORANGE='#c2410c';GRAY='#475569';INK='#172033';LIGHT='#f1f5f9'
OUT=Path(__file__).parent/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})

def canvas(title,subtitle,size=(13,7.5)):
    fig=plt.figure(figsize=size)
    fig.suptitle(title,x=.055,y=.965,ha='left',fontsize=21,fontweight='bold',color=INK)
    fig.text(.055,.90,subtitle,ha='left',fontsize=12,color=GRAY)
    return fig

def diagram(fig,rect=(.04,.10,.92,.74)):
    ax=fig.add_axes(rect);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');return ax

def box(ax,x,y,w,h,text,color=BLUE,fontsize=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.018',
                 facecolor=LIGHT,edgecolor=color,linewidth=1.7))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,color=INK,linespacing=1.65)

def arrow(ax,a,b,label=None):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.6))
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.035,label,ha='center',fontsize=10,color=GRAY)

def save(fig,name):
    OUT.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=160)
    plt.close(fig)


def pipeline():
    fig=canvas('01 · Một corpus, hai đường truy xuất','Ví dụ riêng của bài: từ khóa BM25 và vector LSA; chưa có bước sinh câu trả lời.')
    ax=diagram(fig)
    box(ax,.02,.76,.20,.18,'Tài liệu + ID\nscope + version')
    box(ax,.33,.76,.63,.18,'Fit corpus: analyzer → postings; TF-IDF → SVD\nLưu vocabulary, IDF, components và vector',BLUE)
    arrow(ax,(.22,.85),(.31,.85))
    box(ax,.02,.30,.20,.20,'Query mới\ncùng analyzer')
    box(ax,.33,.30,.25,.20,'Term query; TF-IDF/LSA\nchỉ ở nhánh vector',GREEN)
    box(ax,.69,.30,.27,.20,'BM25 / cosine\n→ lọc → top ứng viên',GREEN)
    arrow(ax,(.22,.40),(.31,.40));arrow(ax,(.58,.40),(.67,.40))
    arrow(ax,(.455,.75),(.455,.52),'dùng lại');arrow(ax,(.825,.75),(.825,.52),'dùng lại')
    box(ax,.69,.02,.27,.16,'Gộp RRF → top-k\nID + score + tựa đề',ORANGE)
    arrow(ax,(.825,.29),(.825,.20))
    ax.text(.02,.08,'Fit chỉ từ corpus; query không refit SVD.\nScore truy xuất không phải xác suất câu trả lời đúng.',fontsize=11,color=GRAY,linespacing=1.8)
    save(fig,'01-retrieval-pipeline')


def hand_scores(hand):
    fig=canvas('02 · TF-IDF và BM25: cùng thứ hạng, khác thang điểm','Corpus: A="api timeout timeout", B="api token", C="token". Query: "api timeout".')
    for rect,key,title,color in [([.08,.26,.36,.55],'tfidf_cosine','TF-IDF + cosine',BLUE),([.58,.26,.36,.55],'bm25','BM25 (k1=1.2, b=0.75)',GREEN)]:
        ax=fig.add_axes(rect);values=hand[key]
        ax.bar(['A','B','C'],values,color=color,width=.55);ax.set_title(title,pad=15)
        ax.set_ylim(0,max(values)*1.22);ax.set_ylabel('Score theo công thức của nhánh')
        for i,v in enumerate(values):ax.text(i,v+max(values)*.04,f'{v:.6f}',ha='center',fontsize=11)
    fig.text(.08,.13,'IDF TF-IDF: api = token ≈ 1.287682; timeout ≈ 1.693147.\nĐộ dài trung bình = 2 term. C không khớp term nào nên không được pad vào kết quả.',color=GRAY,linespacing=1.7)
    save(fig,'02-tfidf-bm25')


def cosine():
    fig=canvas('03 · Cosine so hướng; dot còn phụ thuộc độ dài','Ví dụ hình học tự tạo: q=[1,1], d1=[1,0], d2=[0,1], d3=[2,2].')
    ax=fig.add_axes([.06,.20,.40,.63]);ax.set_aspect('equal');ax.set(xlim=(-.2,2.6),ylim=(-.2,2.6),xlabel='Chiều 1',ylabel='Chiều 2')
    for v,label,c,offset in [(np.array([2,2]),'d3',GREEN,(.08,.08)),(np.array([1,1]),'q',ORANGE,(.08,-.12)),(np.array([1,0]),'d1',BLUE,(.08,.08)),(np.array([0,1]),'d2',BLUE,(.08,.08))]:
        ax.annotate('',xy=v,xytext=(0,0),arrowprops=dict(arrowstyle='->',lw=2.6,color=c));ax.text(*(v+offset),label,color=c)
    ax.grid(alpha=.2)
    right=diagram(fig,(.53,.20,.42,.63))
    box(right,.02,.49,.94,.43,'dot(q,d1) = 1      cosine = 0.7071\ndot(q,d2) = 1      cosine = 0.7071\ndot(q,d3) = 4      cosine = 1.0000',GREEN)
    right.text(.04,.29,'Nhân d3 với 10: dot tăng 10 lần;\ncosine giữ nguyên vì hướng không đổi.',fontsize=12,color=INK,linespacing=1.8)
    right.text(.04,.04,'Với vector đơn vị:\n‖q − d‖² = 2 − 2 cosine(q,d)',fontsize=14,color=ORANGE,linespacing=1.7)
    save(fig,'03-cosine-geometry')


def lsa(index):
    fig=canvas('04 · Từ document–term sang vector LSA','Ma trận thực từ 20 tài liệu của lab. Màu là tọa độ/trọng số; không gán tên ngữ nghĩa cho chiều LSA.')
    ax=fig.add_axes([.07,.24,.40,.54]);im=ax.imshow(index.tfidf,aspect='auto',cmap='Blues',interpolation='nearest')
    ax.set(xlabel=f'Term ID trong vocabulary ({len(index.vocab)} chiều)',ylabel='Document row (0–19)',title='X: TF-IDF đã normalize [20, 280]')
    fig.colorbar(im,ax=ax,fraction=.03,pad=.02)
    ax2=fig.add_axes([.65,.24,.24,.54]);im2=ax2.imshow(index.dense,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
    ax2.set(xlabel='Chiều LSA (0–5)',title='Z: vector dày [20, 6]');fig.colorbar(im2,ax=ax2,fraction=.05,pad=.03)
    fig.text(.56,.51,'× Cᵀ\nchuẩn hóa\n→',ha='center',fontsize=12,color=GREEN,linespacing=1.7)
    fig.text(.07,.10,'C = 6 hàng đầu của Vh trong SVD(X). Query mới: cùng TF-IDF → cùng C → normalize.\nKhông fit trên query/qrels; đây là LSA từ corpus, chưa là neural embedding pretrained.',color=GRAY,linespacing=1.7)
    save(fig,'04-lsa-projection')


def fusion(hand):
    fig=canvas('05 · Reciprocal Rank Fusion','Ví dụ tính tay dùng c=10; cấu hình lab chính dùng c=60 và depth=6 mỗi nhánh.')
    ax=diagram(fig,(.05,.39,.9,.42))
    box(ax,.02,.40,.27,.48,'Từ khóa\nA → B → C',BLUE)
    box(ax,.36,.40,.27,.48,'Vector\nB → C → A',GREEN)
    box(ax,.72,.40,.25,.48,'RRF\nB → A → C',ORANGE)
    arrow(ax,(.63,.63),(.70,.63))
    ax.plot([.155,.155,.68,.68],[.39,.25,.25,.48],color=GRAY,lw=1.6)
    arrow(ax,(.68,.48),(.70,.48))
    ax.text(.02,.06,'RRF(d) = tổng 1 / (c + rank)       ·       rank bắt đầu từ 1',fontsize=14,color=INK)
    ax2=fig.add_axes([.16,.16,.70,.21]);labels=[i for i,_ in hand['rrf']];values=[v for _,v in hand['rrf']]
    ax2.barh(labels[::-1],values[::-1],color=[GRAY,BLUE,ORANGE],height=.6);ax2.set_xlim(0,.21);ax2.set_xlabel('Điểm gộp từ thứ hạng; không phải xác suất')
    for i,v in enumerate(values[::-1]):ax2.text(v+.003,i,f'{v:.6f}',va='center',fontsize=11)
    save(fig,'05-rrf')


def evaluation(report):
    agg=report['aggregates']['test'];methods=['tfidf','bm25','lsa','hybrid'];colors=[BLUE,GREEN,GRAY,ORANGE]
    fig=canvas('06 · Một metric không kể hết câu chuyện','Số đo từ run CPU: 10 query test có đáp án, k=3. Corpus tự tạo nhỏ; chưa có kiểm định ý nghĩa thống kê.')
    ax=fig.add_axes([.08,.30,.38,.49]);vals=[agg[m]['ndcg'] for m in methods]
    ax.bar(methods,vals,color=colors,width=.65);ax.set_ylim(.96,1.003);ax.set_ylabel('NDCG@3 — trục bắt đầu ở 0.96')
    for i,v in enumerate(vals):ax.text(i,v+.0013,f'{v:.4f}',ha='center',fontsize=11)
    ax.set_title('Chênh lệch nhỏ, hybrid chưa thắng')
    ax2=diagram(fig,(.54,.30,.40,.49))
    box(ax2,.02,.47,.94,.48,'Cả bốn phương pháp:\nHit@3 = 1.000 · MRR@3 = 1.000\nRecall@3 = 0.900 · P@3 = 0.433',GREEN,12)
    ax2.text(.04,.16,'Hit=1 chưa bảo đảm tài liệu tốt nhất\nđứng đầu hoặc đã lấy đủ tài liệu đúng.',fontsize=12,color=INK,linespacing=1.7)
    fig.text(.08,.12,'No-answer báo riêng: 2 query; mỗi phương pháp vẫn trả tài liệu ở 1 query.\nKhông cộng no-answer vào trung bình Recall/NDCG trên 10 query có đáp án.',color=GRAY,linespacing=1.8)
    save(fig,'06-evaluation')


def filtering(report):
    fig=canvas('07 · Lọc trước top-k; kiểm tra riêng câu ngoài corpus','Nửa trên là ví dụ thứ hạng giả định. Nửa dưới dùng hai query no-answer đã chạy trong lab.')
    ax=diagram(fig)
    box(ax,.02,.75,.37,.18,'Toàn corpus: P1, R1, P2, P3\nP = public · R = restricted',BLUE,11)
    box(ax,.54,.75,.43,.18,'Lấy top-3 rồi lọc → P1, P2\nLọc trước top-3 → P1, P2, P3',GREEN,11)
    arrow(ax,(.40,.84),(.52,.84))
    box(ax,.02,.27,.44,.31,'Q17: thuật toán quang tử zzz42\nToàn OOV → rỗng\nCả 4 phương pháp không trả ứng viên',BLUE,11)
    box(ax,.53,.27,.44,.31,'Q18: cache trên sao Hỏa\nCó term quen → vẫn trả ứng viên\nNhưng qrels không có tài liệu phù hợp',ORANGE,11)
    ax.text(.02,.07,'Filter demo chưa có auth server-side; IDF/LSA còn dùng cả corpus.\nScore dương không chứng minh có nguồn đáp ứng nhu cầu.',fontsize=12,color=GRAY,linespacing=1.8)
    assert all(v['returned_any']==1 and v['queries']==2 for v in report['no_answer'].values())
    save(fig,'07-filter-no-answer')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    report=json.loads((args.run/'report.json').read_text());index=SearchIndex(documents())
    if report['evaluation']['corpus_hash']!=index.corpus_hash:raise ValueError('Report does not match this corpus')
    pipeline();hand_scores(report['hand']);cosine();lsa(index);fusion(report['hand']);evaluation(report['evaluation']);filtering(report['evaluation'])
    print('Wrote 7 PNG and 7 SVG figures to',OUT)
if __name__=='__main__':main()
