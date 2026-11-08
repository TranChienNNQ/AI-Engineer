import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from data import make_record,CAPTIONS
from media import stft_power

BLUE='#255eaa';GREEN='#20866e';ORANGE='#de8a1d';RED='#bb4955';GRAY='#57677c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})


def box(ax,x,y,w,h,label,color=BLUE,size=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',edgecolor=color,facecolor=color+'12',linewidth=1.8))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)


def arrow(ax,start,end):ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.8,color=GRAY))


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).with_name('figures'));a=p.parse_args()
    r=json.loads(a.report.read_text());a.out.mkdir(parents=True,exist_ok=True);e=r['experiment']
    def save(fig,name):
        fig.savefig(a.out/(name+'.png'),dpi=145,bbox_inches='tight',facecolor='white')
        fig.savefig(a.out/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));fig.suptitle('Biểu diễn từng phương thức và chọn mục tiêu liên kết',fontsize=17)
    for i,(label,encoder,color) in enumerate([('Text / token','Text encoder',BLUE),('Image / pixel','Vision encoder',GREEN),('Audio / sample','Audio encoder',ORANGE)]):
        y=.76-i*.25;box(ax,.03,y,.21,.14,label,color);box(ax,.32,y,.25,.14,encoder,color)
        arrow(ax,(.25,y+.07),(.305,y+.07));arrow(ax,(.585,y+.07),(.69,.48))
    box(ax,.71,.33,.25,.3,'Alignment / fusion\n\nRetrieval · decision\nGeneration cần decoder',RED,size=11)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'01-multimodal-map')

    fig,axes=plt.subplots(1,3,figsize=(12,4.6));fig.suptitle('Patch có thứ tự: không phải cứ lấy liên tiếp bốn pixel đầu',fontsize=16)
    mat=np.arange(16).reshape(4,4);axes[0].imshow(mat,cmap='Blues')
    for (i,j),v in np.ndenumerate(mat):axes[0].text(j,i,str(v),ha='center',va='center',color='white' if v>8 else 'black')
    axes[0].axhline(1.5,color=ORANGE,lw=3);axes[0].axvline(1.5,color=ORANGE,lw=3)
    axes[0].set(title='Một kênh 4×4; P=2',xticks=range(4),yticks=range(4))
    for ax,label in zip(axes[1:],[0,5]):
        ax.imshow(make_record(label,20)['image'],interpolation='nearest');ax.set_title(CAPTIONS[label]+'; 16×16 RGB')
        ax.set(xticks=[],yticks=[])
        for pos in [3.5,7.5,11.5]:ax.axvline(pos,color='white',lw=.8,alpha=.6);ax.axhline(pos,color='white',lw=.8,alpha=.6)
    fig.text(.5,.02,'RGB4×4/P2: 4patch×12value. Ảnh lab16×16/P4: 16patch×48value; chưa là ViT.',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.91));save(fig,'02-image-patches')

    sample=make_record(0,20);power,freq=stft_power(sample['audio']);fig,axes=plt.subplots(2,1,figsize=(11,6),gridspec_kw={'height_ratios':[1,1.5]})
    axes[0].plot(np.arange(800)/8000*1000,sample['audio'],lw=.8,color=BLUE);axes[0].set(xlabel='Thời gian, ms',ylabel='Biên độ',title='Tone tổng hợp lớp đỏ dọc: ~300Hz, envelope tăng')
    # Extent displays frame-start times 0..80ms; bin centers from rfftfreq.
    im=axes[1].imshow(np.log1p(power).T,aspect='auto',origin='lower',extent=[-4,84,-31.25,4031.25],cmap='magma',interpolation='nearest')
    axes[1].set(xlim=(-4,84),ylim=(0,4000),xticks=[0,16,32,48,64,80],xlabel='Frame-start time, ms (11 frame; không center/pad)',ylabel='Tần số bin, Hz',title='log1p(power); Hann128, hop64, sample rate8kHz')
    fig.colorbar(im,ax=axes[1],label='log1p(power), không dB/mel');fig.tight_layout();save(fig,'03-audio-representation')

    fig,axes=plt.subplots(1,2,figsize=(11,4.6));fig.suptitle('Cosine cần denominator và positive set đúng nhiệm vụ',fontsize=16)
    cos=np.array(r['hand']['contrastive']['cosines']);axes[0].imshow(cos,vmin=0,vmax=1,cmap='Blues')
    for (i,j),v in np.ndenumerate(cos):axes[0].text(j,i,f'{v:.1f}',ha='center',va='center',color='white' if v>.7 else 'black',fontsize=18)
    axes[0].set(xticks=[0,1],xticklabels=['text0','text1'],yticks=[0,1],yticklabels=['image0','image1'],title='Cosine / τ.5 → symmetric loss≈.298736')
    positive=np.array([[1,1,0],[1,1,0],[0,0,1]]);axes[1].imshow(positive,cmap='Greens',vmin=0,vmax=1)
    for (i,j),v in np.ndenumerate(positive):axes[1].text(j,i,'positive' if v else 'negative',ha='center',va='center',fontsize=11,color='white' if v else 'black')
    axes[1].set(xticks=range(3),yticks=range(3),title='Labels[0,0,1]: nhiều positive cùng lớp')
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'04-alignment-loss')

    fig,axes=plt.subplots(1,2,figsize=(12,4.7));history=e['history'];epochs=[v['epoch'] for v in history]
    axes[0].plot(epochs,[v['train_loss'] for v in history],color=BLUE,label='Train trước update')
    axes[0].plot(epochs,[v['validation_loss'] for v in history],color=GREEN,label='Validation sau update')
    axes[0].set_yscale('log');axes[0].set(xlabel='Epoch (1 update/epoch)',ylabel='Positive-set loss, log scale',title=f"Best validation epoch{e['best_epoch']}");axes[0].legend(fontsize=10)
    indices=[0,4,8,12,16,20];scores=np.array(e['test']['image_text_scores'])[indices]
    im=axes[1].imshow(scores,cmap='coolwarm',vmin=-1,vmax=1)
    axes[1].set(xticks=range(6),yticks=range(6),xticklabels=CAPTIONS,yticklabels=CAPTIONS,title='6 test images → 6 canonical captions; cosine')
    axes[1].tick_params(axis='x',rotation=40)
    fig.colorbar(im,ax=axes[1],shrink=.8);fig.tight_layout();save(fig,'05-training-alignment')

    pert=e['test']['perturb'];keys=['image_only','audio_only','matched_fusion','mismatched_fusion','black_image','silent_audio']
    labels=['Image\nonly','Audio\nonly','Matched\nfusion','Mispaired\nfusion','Black\nimage','Silent\naudio'];values=[pert[k]['correct'] for k in keys]
    fig,ax=plt.subplots(figsize=(11,5));ax.bar(labels,values,color=[BLUE,ORANGE,GREEN,RED,GRAY,GRAY]);ax.set(ylim=(0,28),ylabel='Đúng / 24 test queries',title='Thiếu nhánh được bỏ khỏi fusion; tín hiệu hỏng/sai cặp là tình huống khác')
    for i,v in enumerate(values):ax.text(i,v+.5,str(v),ha='center',fontsize=14)
    fig.text(.5,.01,'Gallery: 6 canonical captions. Audio roll4 ở fixture mispaired; không là benchmark corruption ngoài đời.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,'06-fusion-faults')

    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));fig.suptitle('Đánh giá từ media hợp lệ đến claim có bằng chứng',fontsize=17)
    rows=[('Ingest / preprocess','Format, rate, size, duration và version',BLUE),('Pair / align','Event, source, timestamp, region và quyền',GREEN),('Task / generation','Retrieval, fusion, ASR/VQA hoặc decoder',ORANGE),('Evaluate / trace','Grounding, missing/conflict, quality và latency',RED)]
    for i,(label,note,color) in enumerate(rows):
        y=.78-i*.23;box(ax,.06,y,.35,.14,label,color);ax.text(.48,y+.07,note,va='center',fontsize=12)
        if i<3:arrow(ax,(.235,y-.005),(.235,y-.075))
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'07-evaluation-workflow')
    print(json.dumps(dict(png=7,svg=7,out=str(a.out))))


if __name__=='__main__':main()
