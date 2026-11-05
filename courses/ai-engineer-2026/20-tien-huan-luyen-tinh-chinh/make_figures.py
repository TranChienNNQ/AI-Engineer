"""Six original lesson figures, from formulas and measured stage artifacts."""
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
def box(ax,x,y,text,w=.25,h=.18,color=BLUE):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.012',ec=color,fc=color+'12'))
    ax.text(x,y,text,ha='center',va='center',fontsize=11)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':GRAY,'lw':1.5})

def stages():
    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    for x,text in [(.15,'Random weight'),(.5,'Pretraining LM\n→ base checkpoint'),(.85,'SFT prompt–answer\n→ model tác vụ')]:box(ax,x,.65,text)
    arrow(ax,(.29,.65),(.36,.65));arrow(ax,(.64,.65),(.71,.65))
    box(ax,.5,.25,'CPT: tiếp tục LM miền\n(lựa chọn chưa chạy trong lab)',w=.40,h=.19,color=TEAL)
    arrow(ax,(.5,.55),(.5,.36));arrow(ax,(.71,.25),(.85,.54))
    ax.set_title('Học phân phối văn bản khác với học hành vi theo yêu cầu',weight='bold',fontsize=15)
    ax.text(.5,.035,'Lab: pretraining nhỏ từ đầu → SFT; không có pretrained LLM tải ngoài hoặc RLHF.',ha='center',color=GRAY)
    save(fig,'01-stages')

def masks():
    fig,ax=plt.subplots(figsize=(12,5));ax.axis('off')
    table=ax.table(cellText=[['0','1','2','3','4'],['BOS','p₁','p₂','a₁','a₂'],['IGNORE','IGNORE','a₁','a₂','EOS'],['0','0','1','1','1']],
        rowLabels=['index','input x','label y','chấm loss?'],cellLoc='center',bbox=[.12,.35,.85,.43])
    table.auto_set_font_size(False);table.set_fontsize(13)
    for col in range(5):table[(2,col)].set_facecolor('#ccfbf1' if col>=2 else '#e2e8f0')
    ax.set_title('Prompt dài P=2: mask y[:2], không mask mất answer đầu',weight='bold',fontsize=16)
    ax.text(.5,.16,'Query tại p₂ dự đoán a₁. EOS được chấm; PAD bị bỏ.\nPrompt vẫn là context và có thể nhận gradient qua attention.',ha='center',va='center',color=GRAY,linespacing=1.7)
    save(fig,'02-loss-mask')

def trainable():
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    counts=[33699,10659];frozen=[0,23040]
    axes[0].bar(['Full SFT','Head-only'],counts,label='Trainable',color=TEAL)
    axes[0].bar(['Full SFT','Head-only'],frozen,bottom=counts,label='Frozen',color='#cbd5e1')
    for i,n in enumerate(counts):axes[0].text(i,n/2,f'{n:,}',ha='center',color='white',weight='bold')
    axes[0].set(ylabel='Parameter',title='Model lab: 33,699 parameter');axes[0].legend()
    axes[1].axis('off');axes[1].text(.02,.9,'LoRA: W frozen, học A và B\n\ny = Wx + (α/r) BAx\n\nW: m × n,  A: r × n,  B: m × r\n\nSố parameter cập nhật: r(m+n)\n\nChỉ demo số/gradient, chưa train adapter.',va='top',fontsize=13)
    fig.suptitle('Số weight được cập nhật và mục tiêu học là hai lựa chọn riêng',weight='bold',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'03-trainable')

def splits():
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    bars=axes[0].bar(['Train','Validation','Test'],[32,8,8],color=[BLUE,TEAL,ORANGE])
    axes[0].bar_label(bars,labels=['32 nhóm / 192 mẫu','8 nhóm / 48 mẫu','8 nhóm / 48 mẫu'],padding=5,fontsize=10)
    axes[0].set(ylim=(0,39),ylabel='Nhóm service–operation',title='48 nhóm, không giao nhau giữa split')
    axes[1].axis('off');axes[1].text(.02,.92,'Tokenizer fit: chỉ LM train\nPretrain: chỉ LM train\nSFT: chỉ prompt–answer train\n\nMỗi stage: 25 × 12 = 300 update\n1 pretrain + 3 nhánh SFT = 1,200\n\nFull có thêm pretrain compute;\nkhông cùng tổng compute với scratch.',va='top',fontsize=12,linespacing=1.5)
    fig.suptitle('Split giữ xuyên các stage, ngân sách cần được báo rõ',weight='bold',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'04-data-budget')

def results(run,report):
    fig,axes=plt.subplots(2,2,figsize=(12,8));stages=report['stages']
    rows=json.loads((run/'base-metrics.json').read_text())
    axes[0,0].plot([r['epoch'] for r in rows],[r['train_ce'] for r in rows],label='Train')
    axes[0,0].plot([r['epoch'] for r in rows],[r['val_ce'] for r in rows],label='Validation')
    axes[0,0].axvline(stages['base']['epoch'],ls='--',color=TEAL,label='Best base')
    axes[0,0].set(title='Pretrain: CE toàn văn bản',xlabel='Epoch',ylabel='Nats/token');axes[0,0].legend(fontsize=9)
    for name in ['full-sft','scratch-sft','head-sft']:
        rows=json.loads((run/f'{name}-metrics.json').read_text())
        axes[0,1].plot([r['epoch'] for r in rows],[r['val_ce'] for r in rows],label=name)
    axes[0,1].set(title='SFT: validation CE answer + EOS',xlabel='Epoch',ylabel='Nats/target token');axes[0,1].legend(fontsize=9)
    names=['base','full-sft','scratch-sft','head-sft'];display=['Base','Full','Scratch','Head-only']
    vals=[100*stages[n]['task_test']['exact'] for n in names]
    bars=axes[1,0].bar(display,vals,color=[GRAY,TEAL,BLUE,ORANGE]);axes[1,0].bar_label(bars,fmt='%.1f%%',padding=4)
    axes[1,0].axhline(100/3,ls='--',color=ORANGE,label='Majority 33.3%');axes[1,0].axhline(100,ls=':',color=GRAY,label='Rule 100%')
    axes[1,0].set(title='Greedy test exact, gồm EOS',ylabel='%',ylim=(0,120));axes[1,0].legend(fontsize=8)
    vals=[stages[n]['lm_test']['ce'] for n in names]
    bars=axes[1,1].bar(display,vals,color=[GRAY,TEAL,BLUE,ORANGE]);axes[1,1].bar_label(bars,fmt='%.3f',padding=4)
    axes[1,1].set(title='LM test CE cũ: thấp hơn là tốt hơn',ylabel='Nats/token',ylim=(0,max(vals)*1.2))
    fig.suptitle('Task mới tốt hơn có thể đi kèm regression ở LM cũ; một seed, miền hẹp',weight='bold',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94),h_pad=2);save(fig,'05-results')

def confusion(report):
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,name in zip(axes.flat,['base','full-sft','scratch-sft','head-sft']):
        matrix=np.array(report['stages'][name]['task_test']['confusion'])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=16)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>8 else 'black')
        ax.set_xticks(range(4),['AUTH','DATA','NET','INVALID']);ax.set_yticks(range(3),['AUTH','DATA','NET'])
        ax.set(xlabel='Nhãn sinh',ylabel='Nhãn thật',title=name)
    fig.suptitle('Không strip/sửa output để che lỗi: giữ cả cột INVALID',weight='bold',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.94),h_pad=2);save(fig,'06-confusion')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    report=json.loads((args.run/'report.json').read_text())
    stages();masks();trainable();splits();results(args.run,report);confusion(report)
    print('Saved 6 PNG + 6 SVG')
if __name__=='__main__':main()
