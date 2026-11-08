"""Seven figures from actual CPU SFT/adapter experiment and labeled arithmetic."""
import argparse
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


def canvas(title,subtitle):
    fig,ax=plt.subplots(figsize=(13,7.5));ax.set(xlim=(0,13),ylim=(0,7));ax.axis('off')
    fig.suptitle(title,x=.055,y=.96,ha='left',fontsize=18,fontweight='bold',color='#172d45')
    fig.text(.055,.89,subtitle,fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.04,right=.98,bottom=.08,top=.84)
    return fig,ax


def box(ax,x,y,w,h,label,color=BLUE,fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.12',
                              facecolor=color+'12',edgecolor=color,linewidth=1.5))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color='#172d45',linespacing=1.6)


def arrow(ax,x,y,xx,yy,color=GRAY):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=15,color=color,lw=1.5))


def save(fig,name,out):
    for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=170,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'figures');args=p.parse_args()
    r=json.loads(args.report.read_text());out=args.out;out.mkdir(parents=True,exist_ok=True);e=r['experiment']
    fig,ax=canvas('01 · SFT, LoRA và QLoRA nằm trên các trục khác nhau',
                  'SFT định nghĩa target/loss; adapter định nghĩa trainables; quantization định nghĩa representation của base.')
    for x,title,body,color in [(.5,'OBJECTIVE','SFT: tăng xác suất\nanswer mẫu + EOS',BLUE),
                               (4.9,'TRAINABLES','Full weights hoặc\nLoRA A/B nhỏ',GREEN),
                               (9.3,'BASE STORAGE','FP32/BF16 hoặc\npacked 4-bit',GOLD)]:
        box(ax,x,3.8,3.2,1.8,title+'\n'+body,color,10)
    box(ax,1.1,1.2,10.8,1.25,'Cùng objective SFT: full fine-tuning → LoRA → LoRA trên frozen quantized base',GRAY,11)
    for x in [2.1,6.5,10.9]:arrow(ax,x,3.7,x,2.6)
    ax.text(6.5,.35,'CPU NF4 teaching branch không được coi là đã chạy bitsandbytes QLoRA/GPU.',ha='center',fontsize=11,color=RED)
    save(fig,'01-three-axes',out)

    fig,ax=canvas('02 · Mask target của prompt, vẫn giữ prompt làm context',
                  'Ví dụ token-level đơn giản; input/label shift đúng một lần, không lấy string length làm token boundary.')
    tokens=['BOS','USER','p1','p2','ASSISTANT','NET','EOS'];xs=[.5+1.8*i for i in range(7)]
    for x,token in zip(xs,tokens):box(ax,x,4.9,1.5,.8,token,BLUE,9)
    labels=['IGNORE','IGNORE','IGNORE','IGNORE','NET','EOS']
    for i,(x,label) in enumerate(zip(xs,labels)):
        box(ax,x,2.7,1.5,.8,label,GRAY if i<4 else GREEN,9)
        arrow(ax,x+.75,4.8,x+.75,3.6,GRAY if i<4 else GREEN)
    ax.text(.5,6.15,'Full sequence: BOS USER p1 p2 ASSISTANT NET EOS',fontsize=11)
    ax.text(.5,1.65,'Logit tại ASSISTANT dự đoán NET; logit tại NET dự đoán EOS.',fontsize=12,color=GREEN)
    ax.text(.5,.7,'IGNORE: direct logit gradient0. Prompt representation vẫn có thể nhận indirect gradient.',fontsize=11,color=RED)
    save(fig,'02-sft-mask',out)

    fig,ax=canvas('03 · LoRA: giữ base, học update qua A và B',
                  'Convention column vector: y=W0x + (alpha/r) B(Ax). Ví dụ rank1/alpha2 của bài.')
    box(ax,.4,3.6,1.7,1.2,'x\n[2,1,3]',GRAY,11)
    box(ax,3.3,4.9,3,1,'W0 frozen (2×3)\nW0x=[−1,5]',BLUE,10)
    box(ax,3.3,2.3,2.2,1,'A (1×3)\nAx=5',GREEN,10)
    box(ax,6.6,2.3,2.7,1,'B (2×1), scale2\nupdate=[2,−1]',GREEN,10)
    box(ax,10.4,3.6,2.1,1.2,'SUM\ny=[1,4]',GOLD,11)
    arrow(ax,2.2,4.2,3.2,5.4);arrow(ax,2.2,4.2,3.2,2.8)
    arrow(ax,5.6,2.8,6.5,2.8);arrow(ax,6.4,5.4,10.3,4.3);arrow(ax,9.4,2.8,10.3,4)
    ax.text(6.5,.85,'Target[1,2], loss2. gradB=[0,20]ᵀ; gradA=[−.8,−.4,−1.2]. Base không cập nhật.',ha='center',fontsize=11)
    save(fig,'03-lora-path',out)

    q=r['quantization'];fig,ax=canvas('04 · NF4: codebook + block scale + packed nibbles',
                  'Local CPU format: block32, FP32 scale; hình dùng block8 để nhìn padding. Không có DQ/paged kernels.')
    box(ax,.4,4.5,4,1.2,'BLOCK WEIGHTS\n[−1,−.5,0,.5,1,.2] + pad0/0',BLUE,10)
    box(ax,5.2,4.5,3.2,1.2,'SCALE = absmax = 1\nNearest NF4 level',GOLD,10)
    box(ax,9.2,4.5,3.2,1.2,'CODES 0..15\n'+str(q['codes']),GREEN,9)
    arrow(ax,4.5,5.1,5.1,5.1);arrow(ax,8.5,5.1,9.1,5.1)
    box(ax,.7,1.8,5.3,1.3,'Hai codes [0,2] → byte0x20=32\nlow nibble0 / high nibble2',GREEN,11)
    box(ax,7,1.8,5.3,1.3,'Unpack → lookup × scale → FP32 matmul\nGiữ grad theo input, không học codes',BLUE,10)
    arrow(ax,10.8,4.4,9.65,3.2)
    ax.text(6.5,.65,'NF4 không có các mức cách đều như uniform INT4; quantized base khác original FP32 base.',ha='center',fontsize=11,color=RED)
    save(fig,'04-nf4-storage',out)

    names=['full-sft','lora','packed-nf4-lora'];labels=['Full SFT','LoRA FP32','Packed NF4 + LoRA']
    fig,axes=plt.subplots(1,2,figsize=(13,7.5));fig.subplots_adjust(left=.09,right=.97,bottom=.29,top=.75,wspace=.3)
    for ax,key,title,unit in [(axes[0],'trainable','Parameter cập nhật','Parameters'),(axes[1],'resident_state_bytes','Static model state','Bytes')]:
        values=[e['stages'][name][key] for name in names];ax.bar(range(3),values,color=[BLUE,GREEN,GOLD])
        for i,v in enumerate(values):ax.text(i,v+max(values)*.03,f'{v:,}',ha='center',fontsize=10)
        ax.set_xticks(range(3),labels,fontsize=9);ax.set_ylim(0,max(values)*1.25);ax.set_ylabel(unit);ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('05 · Ít trainables khác ít resident weights',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Số từ experiment CPU: LoRA giữ base rồi thêm adapter; nhánh quantized chỉ quantize linear base weights.',fontsize=10,color=GRAY)
    fig.text(.055,.14,'Static state chưa có gradients/optimizer/activations/dequant temporaries/global codebook; không là peak VRAM.',fontsize=11,color=RED)
    fig.text(.055,.06,'Cùng base ancestor 6.960 params; adapter r4/alpha8 và5 targets có1.176 trainable params.',fontsize=11,color=GRAY)
    save(fig,'05-memory-and-parameters',out)

    fig,axes=plt.subplots(1,2,figsize=(13,7.5));fig.subplots_adjust(left=.08,right=.96,bottom=.29,top=.75,wspace=.28)
    ax=axes[0]
    for name,label,color in zip(names,labels,[BLUE,GREEN,GOLD]):
        history=e['stages'][name]['history'];ax.plot([x['epoch'] for x in history],[x['validation_ce'] for x in history],label=label,color=color)
    ax.set_yscale('log');ax.set(xlabel='SFT epoch',ylabel='Validation answer CE (log)');ax.legend(fontsize=9);ax.grid(alpha=.15)
    ax=axes[1];all_names=['base']+names;heights=[e['stages'][name]['lm_test']['ce'] for name in all_names]
    ax.bar(range(4),heights,color=[GRAY,BLUE,GREEN,GOLD]);ax.set_xticks(range(4),['Base','Full','LoRA','NF4+LoRA']);ax.set_ylim(0,max(heights)*1.25)
    for i,v in enumerate(heights):ax.text(i,v+.2,f'{v:.3f}',ha='center',fontsize=10)
    ax.set_ylabel('Old-text test CE; thấp hơn tốt hơn');ax.set_title('Frozen base chưa bảo đảm retention')
    fig.suptitle('06 · Học task mới và giữ khả năng cũ là hai phép kiểm',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Actual tiny CPU run, một seed. Các nhánh đều24/24 task test; rule lookup cũng24/24.',fontsize=10,color=GRAY)
    fig.text(.055,.14,'Split chung events/rule; LR full và adapters khác nhau. Không là benchmark QLoRA/GPU hoặc ưu thế tổng quát.',fontsize=11,color=RED)
    fig.text(.055,.06,'Best chọn bằng validation trước test. Adapter bật có thể làm old-text CE tăng dù base không đổi.',fontsize=11,color=GRAY)
    save(fig,'06-training-and-retention',out)

    fig,ax=canvas('07 · Từ dữ liệu SFT đến artifact có thể kiểm tra',
                  'Core đã train CPU; PEFT Linear CPU đã kiểm. CUDA QLoRA recipe chưa được chạy trên máy biên soạn.')
    top=[('PIN DATA + BASE','Group split / identity'),('CHECK MASK + TARGETS','Shift / EOS / trainables'),('TRAIN + SELECT','Validation / frozen checks')]
    bottom=[('RELEASE + OBSERVE','Quality / cost / rollback'),('SAVE / LOAD / MERGE','Adapter identity / parity'),('EVALUATE','Task + retention + format')]
    for i,(a,b) in enumerate(top):box(ax,.45+4.4*i,4.4,3.3,1.2,a+'\n'+b,BLUE,9)
    for i,(a,b) in enumerate(bottom):box(ax,.45+4.4*i,1.7,3.3,1.2,a+'\n'+b,GREEN,9)
    arrow(ax,3.85,5,4.75,5);arrow(ax,8.25,5,9.15,5);arrow(ax,10.9,4.3,10.9,3)
    arrow(ax,9.15,2.3,8.25,2.3);arrow(ax,4.75,2.3,3.85,2.3)
    ax.text(6.5,.45,f"{r['evaluation']['passed']}/{r['evaluation']['checks']} mechanism checks; packed NF4 CPU không thay bằng chứng GPU/LLM integration.",ha='center',fontsize=11)
    save(fig,'07-adapter-release',out)


if __name__=='__main__':main()
