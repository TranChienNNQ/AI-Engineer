"""Six original static figures from formulas and a measured run; PNG + SVG."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','figure.facecolor':'white'})
BLUE,TEAL,ORANGE='#2563eb','#0f766e','#c2410c'

def save(fig,name):
    for ext in ['png','svg']: fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)

def diagram(title,labels,name,footer):
    fig,ax=plt.subplots(figsize=(12,5)); ax.axis('off'); ax.set(xlim=(0,1),ylim=(0,1))
    n=len(labels)
    for i,label in enumerate(labels):
        x=.05+i*.9/n; width=.9/n-.025
        ax.add_patch(FancyBboxPatch((x,.38),width,.28,boxstyle='round,pad=.01',ec=BLUE,fc='#eff6ff'))
        ax.text(x+width/2,.52,label,ha='center',va='center',fontsize=12)
        if i<n-1: ax.annotate('',xy=(x+width+.02,.52),xytext=(x+width,.52),arrowprops={'arrowstyle':'->','color':TEAL})
    ax.set_title(title,weight='bold',fontsize=16)
    ax.text(.5,.12,footer,ha='center',va='center',color='#475569',fontsize=11)
    save(fig,name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    diagram('Văn bản và mô hình là hai lớp cần nối đúng hợp đồng',
            ['Văn bản\nAPI lỗi + emoji','Tokenizer\nbyte → ID','Embedding\n+ Transformer','Head\nlogits → ID','Decode\nID → byte → chữ'],
            '01-pipeline','Tokenizer không truy cập production; xác suất cao không tự là bằng chứng sự thật.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,title,rows in zip(axes,['Dạng dựng sẵn','Dạng tổ hợp'],[
        [['Hiển thị','é'],['Code point','U+00E9'],['UTF-8 (decimal)','195, 169'],['Số code point / byte','1 / 2']],
        [['Hiển thị','e + dấu sắc'],['Code point','U+0065, U+0301'],['UTF-8 (decimal)','101, 204, 129'],['Số code point / byte','2 / 3']]]):
        ax.axis('off');ax.set_title(title,weight='bold')
        table=ax.table(cellText=rows,cellLoc='left',colWidths=[.48,.52],bbox=[0,.15,1,.7]);table.auto_set_font_size(False);table.set_fontsize(11)
    fig.suptitle('Nhìn giống nhau chưa chắc có cùng chuỗi code point hoặc byte',weight='bold',fontsize=16)
    fig.text(.5,.025,'Lab giữ nguyên input, không normalize; token count phụ thuộc tokenizer.',ha='center',color=TEAL)
    save(fig,'02-unicode')
    diagram('BPE tính tay: corpus aba × 2, ac × 1',
            ['Base byte IDs\na=100, b=101\nc=102','Merge 1\n(a,b) → 259\nID 259 = ab','Merge 2\n(259,a) → 260\nID 260 = aba','Encode mới\nabac\n[260,102]'],
            '03-bpe','Tần suất hòa: chọn tuple ID nhỏ nhất. Vocabulary = 259 + 2 = 261; giữ byte cũ.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    probabilities=[.5,.25,.125]
    bars=axes[0].bar(['A | BOS','B | BOS,A','EOS | BOS,A,B'],probabilities,color=[BLUE,TEAL,ORANGE])
    axes[0].bar_label(bars,labels=['1/2','1/4','1/8'],padding=4);axes[0].set(ylim=(0,.65),ylabel='Xác suất nhãn đúng',title='Xác suất chuỗi = tích = 1/64')
    axes[1].axis('off');axes[1].text(.08,.85,'NLL tổng = ln64 ≈ 4.158883\n\n3 nhãn: A, B, EOS (không chấm BOS)\n\nCE = ln64 / 3 = ln4 ≈ 1.386294\n\nPPL = exp(CE) = 4',va='top',fontsize=14)
    fig.suptitle('Perplexity là biến đổi của NLL trung bình, không phải accuracy',weight='bold',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'04-likelihood')
    diagram('Shift đúng cho một tài liệu: nội dung IDs [10,11]',
            ['Full sequence\n[BOS,10,11,EOS]','Input x\n[BOS,10,11]','Labels y\n[10,11,EOS]','Loss\nchấm 3 nhãn\nPAD → IGNORE'],
            '05-shift','Causal mask chặn future; loss mask không thay thế attention mask.')
    report=json.loads((args.run/'report.json').read_text());rows=json.loads((args.run/'metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].plot([r['epoch'] for r in rows],[r['train_ce'] for r in rows],label='Train: trong cập nhật')
    axes[0].plot([r['epoch'] for r in rows],[r['val_ce'] for r in rows],label='Validation: cuối epoch')
    axes[0].axvline(report['best_epoch'],ls='--',color=TEAL,label=f"Best epoch {report['best_epoch']}")
    axes[0].set(xlabel='Epoch',ylabel='Nats/token',title='Next-token cross-entropy');axes[0].legend(fontsize=9)
    values=[report['baselines']['unigram_add1']['perplexity'],report['baselines']['bigram_add1']['perplexity'],report['test']['perplexity']]
    bars=axes[1].bar(['Unigram +1','Bigram +1','Transformer'],values,color=[ORANGE,BLUE,TEAL]);axes[1].bar_label(bars,fmt='%.3f',padding=4)
    axes[1].set(ylim=(0,max(values)*1.2),ylabel='PPL/token',title='Test: cùng tokenizer, gồm EOS')
    fig.suptitle('Corpus template hẹp: metric tốt vẫn có thể sinh câu lặp hoặc trộn sai',weight='bold',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'06-training')
    print('Saved 6 PNG + 6 SVG')
if __name__=='__main__': main()
