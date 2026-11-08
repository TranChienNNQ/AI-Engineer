"""Recreate all figures from equations and an actual lab report. No external images."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'figures');a=parser.parse_args()
    report=json.loads(a.report.read_text());a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    blue,orange,green,red='#245ca6','#dd8a21','#25836a','#be4b53'
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(a.out/f'{name}.{ext}',dpi=155,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    def box(ax,x,y,text,color=blue,w=.20,h=.20):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.014',fc=color+'14',ec=color,lw=1.5))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
    def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':1.7,'color':'#48576b'})
    fig,ax=plt.subplots(figsize=(12,5.6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.set_title('Từ SFT đến học từ dữ liệu ưu tiên',fontsize=17,pad=18)
    for x,text in [(.03,'SFT policy'),(.29,'Cặp ưu tiên\nchosen / rejected'),(.55,'Reward model\nrφ(x, y)'),(.81,'Tối ưu policy\nreward − β KL')]:box(ax,x,.63,text,w=.16)
    for x in [.19,.45,.71]:arrow(ax,x,.73,x+.10,.73)
    ax.text(.5,.92,'Nhánh RM + tối ưu reward (InstructGPT dùng PPO)',ha='center',color=blue)
    box(ax,.03,.23,'SFT → clone\npolicy + reference',w=.22,color=green)
    box(ax,.39,.23,'Cặp ưu tiên cố định\n+ log-probabilities',w=.22,color=green)
    box(ax,.75,.23,'DPO sigmoid loss\nchỉ cập nhật policy',w=.22,color=green)
    arrow(ax,.25,.33,.39,.33);arrow(ax,.61,.33,.75,.33)
    ax.text(.5,.07,'Lab: bandit liệt kê 3 hành động + TinyLM DPO; chưa chạy PPO hay thu nhãn người thật.',ha='center',fontsize=11)
    save(fig,'01-pipelines')

    beta=np.geomspace(.1,10,250);p=1/(1+np.exp(-np.log(3)/beta));kl=p*np.log(2*p)+(1-p)*np.log(2*(1-p))
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axs[0].semilogx(beta,p,color=blue,lw=2);axs[0].axhline(.5,color='gray',ls='--',label='Reference = .5')
    axs[0].plot(1,.75,'o',color=orange);axs[0].annotate('β=1 → p=.75',(1,.75),xytext=(1.3,.85),arrowprops={'arrowstyle':'->'})
    axs[0].set(ylabel='Xác suất hành động reward cao',xlabel='β (log scale)',ylim=(.45,1.03));axs[0].legend(loc='lower left')
    axs[1].semilogx(beta,kl,color=green,lw=2);axs[1].set(xlabel='β (log scale)',ylabel='KL(policy || reference), nat')
    fig.suptitle('Nghiệm đóng: reference [.5, .5], reward [ln 3, 0]',fontsize=15)
    save(fig,'02-kl-beta')

    m=np.linspace(-5,5,300);fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for b,c in [(.5,blue),(1,green),(2,orange)]:
        axs[0].plot(m,np.logaddexp(0,-b*m),label=f'β={b}',color=c)
        axs[1].plot(m,b/(1+np.exp(b*m)),label=f'β={b}',color=c)
    h=report['hand'];axs[0].plot(h['margin'],h['loss'],'o',color=red)
    axs[0].annotate('Ví dụ tay: L=.562998',(h['margin'],h['loss']),xytext=(.6,3),arrowprops={'arrowstyle':'->'},fontsize=10)
    for ax in axs:ax.set_xlabel('Margin m so với reference');ax.axvline(0,color='gray',ls=':',lw=1);ax.legend()
    axs[0].set_ylabel('Loss = softplus(−βm)');axs[1].set_ylabel('|dL/dm| = β sigmoid(−βm)')
    fig.suptitle('DPO: margin bằng 0 vẫn có gradient',fontsize=15)
    save(fig,'03-dpo-loss')

    fig,ax=plt.subplots(figsize=(12,4.2));ax.axis('off');ax.set_title('Causal shift: chỉ chấm completion, gồm EOS',fontsize=16,pad=24)
    rows=[['BOS','USER','route','timeout','ASSISTANT','NET','CHECK','EOS'],
          ['BOS','USER','route','timeout','ASSISTANT','NET','CHECK','—'],
          ['IGNORE','IGNORE','IGNORE','IGNORE','NET','CHECK','EOS','—']]
    table=ax.table(cellText=rows,rowLabels=['Full sequence','Input','Target'],cellLoc='center',loc='center',colWidths=[.112]*8)
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.3)
    for j in range(8):
        table[(2,j)].set_facecolor('#d5eee5' if 4<=j<=6 else '#eceff4')
    ax.text(.5,.05,'Logit tại ASSISTANT → NET. IGNORE không vào sum logps; prompt vẫn tham gia attention.',ha='center',transform=ax.transAxes,fontsize=11)
    save(fig,'04-token-mask')

    bandit=report['bandit'];rs=bandit['results'];x=np.arange(4);names=['Reference']+[f"β={r['beta']:g}" for r in rs]
    fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    proxies=[bandit['reference_proxy_reward']]+[r['proxy_reward'] for r in rs]
    utilities=[bandit['reference_utility']]+[r['independent_utility'] for r in rs]
    axs[0].bar(x-.18,proxies,.36,label='Proxy reward',color=blue);axs[0].bar(x+.18,utilities,.36,label='Utility riêng',color=orange)
    axs[0].axhline(0,color='gray',lw=.8);axs[0].set_xticks(x,names);axs[0].legend();axs[0].set_ylabel('Kỳ vọng điểm (hai thang riêng)')
    policies=np.array([bandit['reference']]+[r['policy'] for r in rs]);bottom=np.zeros(4)
    for i,(name,c) in enumerate(zip(bandit['actions'],[green,blue,red])):
        axs[1].bar(x,policies[:,i],bottom=bottom,color=c,label=name);bottom+=policies[:,i]
    axs[1].set_xticks(x,names);axs[1].set_ylabel('Xác suất hành động');axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.1),ncol=3,fontsize=9)
    fig.suptitle('RM chưa thấy verbose_false: tăng proxy có thể làm giảm utility',fontsize=14)
    save(fig,'05-reward-hacking')

    e=report['experiment'];history=e['dpo']['history'];fig,axs=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
    for key,c,label in [('train_loss',blue,'Train DPO loss'),('validation_loss',orange,'Validation DPO loss')]:axs[0].plot([r['epoch'] for r in history],[r[key] for r in history],label=label,color=c)
    axs[0].axvline(e['dpo']['best_epoch'],color='gray',ls=':',label='Checkpoint đã chọn');axs[0].set(xlabel='Epoch',ylabel='Mean pair loss');axs[0].legend(fontsize=9)
    xx=np.arange(2);routes=[e['test'][k]['generation']['route_correct'] for k in ['sft','dpo']];qs=[e['test'][k]['generation']['exact_quality'] for k in ['sft','dpo']]
    axs[1].bar(xx-.18,routes,.36,color=blue,label='Route đúng');axs[1].bar(xx+.18,qs,.36,color=green,label='LABEL CHECK EOS')
    axs[1].set_xticks(xx,['SFT','DPO']);axs[1].set(ylim=(0,14),ylabel='Số đúng / 12 prompt test');axs[1].legend(fontsize=9)
    for i in range(2):axs[1].text(i+.18,qs[i]+.3,str(qs[i]),ha='center')
    fig.suptitle('Kết quả TinyLM CPU: dữ liệu tổng hợp, cùng rule/template giữa các split',fontsize=13)
    save(fig,'06-training')

    fig,axs=plt.subplots(1,2,figsize=(11.5,4.3),gridspec_kw={'width_ratios':[1,1.1]},layout='constrained')
    xx=np.arange(3);axs[0].bar(xx-.18,[.4,.3,.3],.36,color=blue,label='Reference');axs[0].bar(xx+.18,[.2,.1,.7],.36,color=orange,label='Policy')
    axs[0].set_xticks(xx,['A (chosen)','B (rejected)','C (khác)']);axs[0].set(ylim=(0,.85),ylabel='Xác suất completion');axs[0].legend()
    axs[1].axis('off');axs[1].text(.04,.85,'Cặp chỉ quan sát A > B',fontsize=14,weight='bold')
    axs[1].text(.04,.62,'m = ln(.2/.4) − ln(.1/.3)\n   = ln 1.5 > 0',fontsize=13,linespacing=1.7)
    axs[1].text(.04,.30,'β=1: loss .693147 → .510826\nNhưng P(A): .4 → .2',fontsize=13,linespacing=1.8,color=red)
    fig.suptitle('Ưu tiên tương đối tốt hơn chưa bảo đảm chosen tăng xác suất',fontsize=14)
    save(fig,'07-relative-absolute')
    print(json.dumps({'figures':7,'formats':['png','svg'],'out':str(a.out)},ensure_ascii=False))


if __name__=='__main__':main()
