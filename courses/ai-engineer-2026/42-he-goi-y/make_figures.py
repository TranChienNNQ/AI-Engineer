import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE='#255eaa';GREEN='#20866e';ORANGE='#de8a1d';RED='#bb4955';GRAY='#57677c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})


def box(ax,x,y,w,h,label,color=BLUE,size=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',edgecolor=color,facecolor=color+'12',linewidth=1.8))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)


def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.8,color=GRAY))


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).with_name('figures'));a=p.parse_args()
    r=json.loads(a.report.read_text());e=r['experiment'];a.out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(a.out/(name+'.png'),dpi=145,bbox_inches='tight',facecolor='white')
        fig.savefig(a.out/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));fig.suptitle('Gợi ý cần dữ liệu, candidate và quyết định cả danh sách',fontsize=17)
    stages=[('Catalog + history','As-of time, quyền và availability',BLUE),('Candidate generation','Thu hẹp tập item; giữ coverage',GREEN),('Scoring / ranking','Chấm relevance theo context',ORANGE),('Re-ranking → display','List policy, diversity và logging',RED)]
    for i,(label,note,color) in enumerate(stages):
        y=.78-i*.23;box(ax,.04,y,.37,.14,label,color);ax.text(.48,y+.07,note,va='center',fontsize=12)
        if i<3:arrow(ax,(.225,y-.005),(.225,y-.075))
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'01-recommendation-pipeline')

    fig,ax=plt.subplots(figsize=(12,4.6));ax.axvspan(-.4,9.7,color=BLUE,alpha=.09);ax.set(xlim=(-.4,12.6),ylim=(0,3),yticks=[])
    ax.hlines(1.4,0,12,color=GRAY,lw=2);ax.scatter(range(8),[1.4]*8,color=BLUE,s=55,label='Train clicks t0…7')
    ax.scatter([10,11],[1.4,1.4],color=[GREEN,ORANGE],s=90,zorder=3)
    ax.text(4.5,2.4,'Fit tại cutoff t<10\nWeights/pop/item similarity frozen sau fit',ha='center',fontsize=13,color=BLUE)
    ax.text(10,1.75,'Validation\nrequest t10',ha='center',color=GREEN)
    ax.text(11,.55,'Test t11\nhistory gồm click t10',ha='center',color=ORANGE)
    ax.annotate('6 item mới release t11\nitem71 retire t11',xy=(11,1.4),xytext=(8,.05),arrowprops=dict(arrowstyle='->',color=GRAY),fontsize=10,ha='center')
    ax.text(12,2.35,'Item78\nrelease t12\nchưa eligible',ha='center',color=RED,fontsize=10)
    ax.set(xlabel='Thời gian rời rạc của generator, không ngày thực',title='Target tại t không được đưa vào history <t');ax.set_xticks(range(13))
    fig.tight_layout();save(fig,'02-temporal-evaluation')

    fig,axes=plt.subplots(1,2,figsize=(11,4.8));delta=np.linspace(-4,5,200)
    axes[0].plot(delta,np.logaddexp(0,-delta),color=BLUE,lw=2);axes[0].scatter([1],[r['hand']['bpr']['loss']],color=ORANGE,s=60)
    axes[0].annotate('Δ=1 → loss≈.313262',xy=(1,r['hand']['bpr']['loss']),xytext=(-.2,2.4),arrowprops=dict(arrowstyle='->'))
    axes[0].set(xlabel='Δ = score positive − score sampled negative',ylabel='softplus(−Δ)',title='Pairwise loss; chưa là NDCG trực tiếp')
    axes[1].axis('off');axes[1].set(xlim=(0,1),ylim=(0,1))
    box(axes[1],.05,.64,.9,.22,'p=[1,0], qi=[1,0], qj=[0,1]',BLUE,size=11)
    arrow(axes[1],(.5,.61),(.5,.4))
    box(axes[1],.05,.1,.9,.27,"SGD lr.1, cùng trạng thái cũ\np′=[1.026894, −.026894]\nqi′=[1.026894,0], qj′=[−.026894,1]",GREEN,size=11)
    fig.tight_layout();save(fig,'03-bpr-optimization')

    results=e['results'];names=list(results);labels=['Pop','Content','Content\n+fresh','Item\ncosine','MF-BPR','Hybrid\nRRF','Hybrid\n+cap']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7));history=e['history'];epochs=[v['epoch'] for v in history]
    axes[0].plot(epochs,[v['pair_loss'] for v in history],color=BLUE,label='Train sampled pair loss')
    axes[0].plot(epochs,[v['validation_ndcg5'] for v in history],color=GREEN,label='Validation NDCG@5')
    axes[0].axvline(e['best_epoch'],ls='--',color=GRAY);axes[0].set(xlabel='Epoch (1 update/epoch)',title=f"Best MF epoch {e['best_epoch']} theo validation",ylim=(0,.8));axes[0].legend(fontsize=9)
    x=np.arange(7);w=.36
    axes[1].bar(x-w/2,[results[n]['overall']['recall'] for n in names],w,color=BLUE,label='Recall@5')
    axes[1].bar(x+w/2,[results[n]['overall']['ndcg'] for n in names],w,color=GREEN,label='NDCG@5')
    axes[1].set(xticks=x,xticklabels=labels,ylim=(0,1),title='54 test queries; exact eligible protocol');axes[1].tick_params(axis='x',labelsize=9);axes[1].legend(fontsize=9)
    fig.tight_layout();save(fig,'04-training-baselines')

    fig,axes=plt.subplots(1,2,figsize=(12,4.7));methods=['content','content_fresh','mf_bpr','hybrid_rrf'];cohorts=['warm','new_item','new_user']
    values=np.array([[results[m]['cohorts'][c]['recall'] for m in methods] for c in cohorts]);im=axes[0].imshow(values,cmap='Blues',vmin=0,vmax=1)
    for (i,j),v in np.ndenumerate(values):axes[0].text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>.6 else 'black')
    axes[0].set(xticks=range(4),xticklabels=['Content','+fresh','MF','RRF'],yticks=range(3),yticklabels=['Warm (36)','New item (12)','New user (6)'],title='Recall@5 theo cohort; denominator khác nhau')
    c=e['candidate_demo'];bars=axes[1].bar(['Candidate\nRecall@10','Final\nRecall@5'],[c['candidate_recall'],c['final_recall']],color=[ORANGE,GREEN],width=.5)
    axes[1].set(ylim=(0,1),title='Popularity top10 → content scoring\nKhông chèn target vào candidate')
    for b,v in zip(bars,[c['candidate_recall'],c['final_recall']]):axes[1].text(b.get_x()+b.get_width()/2,v+.03,f'{v:.4f}\n5/54',ha='center')
    fig.tight_layout();save(fig,'05-candidates-cold-start')

    fig,axes=plt.subplots(1,3,figsize=(12,4.5));selected=['hybrid_rrf','hybrid_diverse'];col=[BLUE,ORANGE]
    for ax,key,title in zip(axes,['recall','ndcg','mean_topics'],['Recall@5','NDCG@5','Mean topics@5']):
        vals=[results[n]['overall'][key] for n in selected];ax.bar(['RRF','RRF\n+topic cap'],vals,color=col,width=.55)
        for i,v in enumerate(vals):ax.text(i,v+max(vals)*.04,f'{v:.3f}',ha='center',fontsize=12)
        ax.set(title=title,ylim=(0,max(vals)*1.25))
    fig.suptitle('Topic cap2: đa dạng tăng, relevance và catalog coverage có thể giảm',fontsize=15)
    fig.text(.5,.01,'Unique displayed: RRF68/77 → cap64/77; không là fairness guarantee.',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.06,1,.91));save(fig,'06-reranking-tradeoff')

    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));fig.suptitle('Serving và feedback phải dùng cùng nghĩa timestamp / version',fontsize=17)
    items=[('Catalog / features','Eligibility, item mapping, as-of history',BLUE),('Model / index / policy','Version match, fallback và budgets',GREEN),('Display / observation','Request, candidates, positions, impressions',ORANGE),('Outcome / evaluation','Delayed join, slices, offline và experiment',RED)]
    for i,(label,note,color) in enumerate(items):
        y=.78-i*.23;box(ax,.04,y,.4,.14,label,color);ax.text(.5,y+.07,note,va='center',fontsize=11)
        if i<3:arrow(ax,(.24,y-.005),(.24,y-.075))
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'07-serving-feedback')
    print(json.dumps(dict(png=7,svg=7,out=str(a.out))))


if __name__=='__main__':main()
