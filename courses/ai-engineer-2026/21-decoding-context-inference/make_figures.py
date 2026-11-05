"""Seven original static figures, using hand formulas and measured correctness checks."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
ROOT=Path(__file__).resolve().parent
BLUE,TEAL,ORANGE,GRAY='#2563eb','#0f766e','#c2410c','#475569'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','figure.facecolor':'white'})

def save(fig,name):
    for ext in ('png','svg'):fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
def box(ax,x,y,text,w=.24,h=.19,color=BLUE):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.01',ec=color,fc=color+'12'))
    ax.text(x,y,text,ha='center',va='center',fontsize=11)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':GRAY,'lw':1.5})

def pipeline():
    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    for x,text in [(.15,'Template + tokenizer\nkiểm context budget'),(.5,'Prefill / decode\nlogits cho token mới'),(.85,'Policy decoding\nchọn một ID')]:box(ax,x,.65,text)
    arrow(ax,(.28,.65),(.37,.65));arrow(ax,(.63,.65),(.72,.65))
    box(ax,.85,.28,'EOS / hết budget?\nDừng + finish_reason',color=TEAL)
    arrow(ax,(.85,.54),(.85,.39))
    ax.plot([.72,.5,.5],[.28,.28,.54],color=ORANGE,lw=1.6)
    ax.text(.5,.15,'Chưa dừng: nối ID → cập nhật cache / prefix → bước tiếp',ha='center',color=ORANGE)
    ax.set_title('Weight giữ nguyên; mỗi bước chọn token làm thay đổi prefix kế tiếp',weight='bold',fontsize=15)
    save(fig,'01-inference-loop')

def distributions(report):
    data=report['hand'];fig,axes=plt.subplots(1,2,figsize=(12,5));x=np.arange(4)
    for offset,name,label in [(-.25,'tau_half','τ=0.5'),(0,'tau1','τ=1'),(.25,'tau2','τ=2')]:
        axes[0].bar(x+offset,data[name],.25,label=label)
    axes[0].set_xticks(x,['A','B','C','D']);axes[0].set(ylim=(0,.85),ylabel='Probability',title='Temperature: đổi độ tập trung');axes[0].legend()
    for offset,name,label in [(-.25,'tau1','Gốc'),(0,'top_k2','k=2 / p=0.7'),(.25,'top_p_08','p=0.8')]:
        axes[1].bar(x+offset,data[name],.25,label=label)
    axes[1].set_xticks(x,['A','B','C','D']);axes[1].set(ylim=(0,.85),title='Lọc tập candidate rồi chuẩn hóa');axes[1].legend(fontsize=9)
    fig.suptitle('Ví dụ tự tạo p=[0.5,0.25,0.15,0.1]; không phải likelihood một từ thực',weight='bold',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.93));save(fig,'02-distributions')

def beam():
    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    box(ax,.1,.5,'BOS',w=.12)
    box(ax,.39,.76,'A: p=0.6',w=.22);box(ax,.39,.25,'B: p=0.4',w=.22,color=TEAL)
    arrow(ax,(.17,.54),(.27,.73));arrow(ax,(.17,.46),(.27,.28))
    box(ax,.77,.76,'X0..X9: mỗi p=0.1\n→ EOS p=1\nP chuỗi = 0.06',w=.36,h=.27)
    box(ax,.77,.25,'X0: p=0.9 → EOS p=1\nP chuỗi = 0.36\nX1 còn lại: P chuỗi = 0.04',w=.36,h=.27,color=TEAL)
    arrow(ax,(.51,.76),(.58,.76));arrow(ax,(.51,.25),(.58,.25))
    ax.set_title('Greedy đi qua A; beam width 2 giữ B rồi chọn B–X0–EOS',weight='bold',fontsize=15)
    ax.text(.5,.025,'Chỉ cây hữu hạn này được đối chiếu exhaustive; beam hữu hạn không bảo đảm global optimum mọi cây.',ha='center',color=GRAY,fontsize=10)
    save(fig,'03-beam')

def context():
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    left=0
    for amount,label,color in [(10,'Template',GRAY),(20,'Câu hỏi',BLUE),(40,'Log',TEAL),(30,'Output reserve',ORANGE)]:
        axes[0].barh(['Budget 100'],[amount],left=left,color=color,label=label)
        axes[0].text(left+amount/2,0,str(amount),ha='center',va='center',color='white',weight='bold');left+=amount
    axes[0].set(xlim=(0,100),xlabel='Token, sau template/tokenizer',title='Ví dụ budget giả định');axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,fontsize=9)
    vals=[128,256,512,1024]
    bars=axes[1].bar(['1024','2048','4096','8192'],vals,color=BLUE);axes[1].bar_label(bars,fmt='%d MiB',padding=4,fontsize=10)
    axes[1].set(ylim=(0,1200),xlabel='T token',ylabel='MiB',title='KV: 32 layer, B=1, 8 KV head, dₕ=128, fp16')
    fig.suptitle('Cache giảm tính lại nhưng vẫn tốn bộ nhớ; không tăng context limit',weight='bold',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.93));save(fig,'04-context-memory')

def masks():
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    values=np.array([[1,1,1,1,0],[1,1,1,1,1]])
    axes[0].imshow(values,cmap='Greens',vmin=0,vmax=1,aspect='auto')
    for (i,j),value in np.ndenumerate(values):axes[0].text(j,i,'T' if value else 'F',ha='center',va='center',color='white' if value else GRAY,fontsize=16)
    axes[0].set_xticks(range(5),range(5));axes[0].set_yticks([0,1],['query abs 3','query abs 4']);axes[0].set(xlabel='Key absolute position',title='Past=3, new=2: bool True cho đọc')
    axes[1].axis('off');axes[1].text(.03,.91,'Q=K=0, V=[1,2,3,4]\nMột query mới tại vị trí 3:\n\nMask offset đúng → đọc cả 4\nOutput = (1+2+3+4)/4 = 2.5\n\nUpper-left causal 1×4 → chỉ key 0\nOutput = 1.0\n\nAPI không tự suy past offset.',va='top',fontsize=13)
    fig.suptitle('Không dùng causal mask không vuông mà quên absolute position',weight='bold',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'05-cache-mask')

def verification(report):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    vals=[report['context']['positions_uncached'],report['context']['positions_cached']]
    bars=axes[0].bar(['Không cache','Có cache'],vals,color=[ORANGE,TEAL]);axes[0].bar_label(bars,fmt='%d',padding=4)
    axes[0].set(ylim=(0,185),ylabel='Số vị trí qua projection / FFN',title='P=16, N=8: đếm, không phải speedup')
    c=report['cache'];axes[1].axis('off')
    axes[1].text(.02,.92,f"Đối chiếu logits CPU float64\n\nPrefill4 + decode từng token:\nmax error = {c['max_abs_error']:.3e}\n\nPrefill3 + chunk5:\nmax error = {c['chunk_max_abs_error']:.3e}\n\nK/V mỗi tensor: [1,4,8,8]\nTổng K+V: {c['kv_bytes']:,} byte",va='top',fontsize=13)
    fig.suptitle('Giữ độ đúng trước khi đo hiệu năng; không chỉ so một argmax',weight='bold',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'06-verification')

def telemetry(report):
    fig,ax=plt.subplots(figsize=(12,4.5));times=[0,120,160,205,250]
    ax.hlines(1,0,250,color=GRAY,lw=2);ax.scatter(times,[1]*5,s=70,color=[GRAY,BLUE,TEAL,TEAL,TEAL],zorder=3)
    labels=['Request\nbắt đầu','Token 1','Token 2','Token 3','Token 4\nhoàn thành']
    for t,label in zip(times,labels):ax.text(t,1.12,label,ha='center',va='bottom')
    ax.annotate('',xy=(120,.78),xytext=(0,.78),arrowprops={'arrowstyle':'<->','color':BLUE});ax.text(60,.64,'TTFT = 120 ms',ha='center',color=BLUE)
    ax.annotate('',xy=(250,.78),xytext=(120,.78),arrowprops={'arrowstyle':'<->','color':TEAL});ax.text(185,.64,'TPOT = 130 / 3 ≈ 43.33 ms',ha='center',color=TEAL)
    ax.set(xlim=(-15,275),ylim=(.4,1.5),xlabel='ms (timestamp giả định)',yticks=[],title='TTFT khác TPOT và end-to-end; đây không phải benchmark')
    for spine in ['top','left','right']:ax.spines[spine].set_visible(False)
    save(fig,'07-observability')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text())
    pipeline();distributions(report);beam();context();masks();verification(report);telemetry(report)
    print('Saved 7 PNG + 7 SVG')
if __name__=='__main__':main()
