"""Static teaching figures; point metrics come from an executed full lab report."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from data import dataset

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':12,'figure.dpi':130,'savefig.dpi':150,'svg.hashsalt':'lesson43'})
BLUE='#2463a6';ORANGE='#df7b2a';GREEN='#25826c';RED='#bf4358';GRAY='#626b78'


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).parent/'figures');a=p.parse_args()
    r=json.loads(a.report.read_text())['experiment'];out=a.out;out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(out/(name+'.png'),bbox_inches='tight',facecolor='white',metadata={'Software':'lesson43'})
        fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white',metadata={'Date':None,'Creator':'lesson43'})
        plt.close(fig)
    d=dataset();s=dataset(False);t=d['t'];c=d['components']
    fig,axes=plt.subplots(3,1,figsize=(12,8),layout='constrained')
    axes[0].plot(t,d['y'],lw=1,color=BLUE,label='Actual: nhánh shift');axes[0].plot(t[364:],s['y'][364:],color=GREEN,lw=1,label='Nhánh ổn định')
    for x in [224,280,364]:axes[0].axvline(x,color=GRAY,ls='--',lw=.9)
    axes[0].set(title='01 · Một chuỗi, hai chế độ tương lai tổng hợp',ylabel='Đơn vị target');axes[0].legend(ncol=2,loc='upper left')
    axes[1].plot(t,c['trend'],label='Trend',color=BLUE);axes[1].plot(t,c['plan_effect'],label='Hiệu ứng plan',color=ORANGE);axes[1].plot(t,c['shock'],label='Cú đổi chế độ',color=RED)
    axes[1].set(ylabel='Thành phần cộng');axes[1].legend(ncol=3)
    ix=np.arange(336,392);axes[2].plot(ix,c['season'][ix],label='Mùa vụ tuần',color=GREEN);axes[2].plot(ix,c['noise'][ix],label='Noise có phụ thuộc',color=GRAY)
    axes[2].axvline(364,color=RED,ls='--');axes[2].set(xlabel='Index ngày (0-based)',ylabel='Thành phần cộng');axes[2].legend(ncol=2)
    fig.suptitle('Thành phần do generator biết; không phải decomposition ước lượng hoặc feature ẩn',fontsize=11)
    save(fig,'01-series-components')

    fig,ax=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    phases=[(0,224,'Fit ban đầu',BLUE),(224,56,'Validation · chọn',ORANGE),(280,84,'Calibration · q',GREEN),(364,56,'Test · đo',RED)]
    for start,width,label,color in phases:
        ax[0].broken_barh([(start,width)],(1.4,.6),facecolors=color,alpha=.85);ax[0].text(start+width/2,1.7,label,ha='center',va='center',color='white',fontsize=10)
    for start in range(223,413,7):
        if start<279:color=ORANGE
        elif start<363:color=GREEN
        else:color=RED
        ax[0].broken_barh([(start+1,7)],(.5,.5),facecolors=color,edgecolors='white')
    ax[0].set(xlim=(-3,423),ylim=(.2,2.4),xticks=[0,224,280,364,420],yticks=[],xlabel='Ranh giới phải exclusive; mỗi ô nhỏ là 7 target')
    ax[0].set_title('02 · Refit tại 280; weights/scaler giữ cố định suốt calibration và test')
    ax[0].annotate('Refit y[:280]',xy=(280,1.35),xytext=(180,1.05),arrowprops={'arrowstyle':'->'},fontsize=10)
    T=363;ax[1].broken_barh([(T-27,28)],(1,.7),facecolors=BLUE);ax[1].broken_barh([(T+1,7)],(1,.7),facecolors=RED)
    ax[1].axvline(T+.5,color=GRAY,ls='--');ax[1].text(T-13,1.35,'28 y đã quan sát',ha='center',color='white');ax[1].text(T+4,1.35,'7 target',ha='center',color='white')
    ax[1].text(353,.5,'Plan/calendar tương lai chỉ dùng nếu available_at ≤ 363',ha='center',fontsize=10)
    ax[1].set(xlim=(332,374),ylim=(.2,2.2),yticks=[],xticks=[336,350,363.5,370],xticklabels=['336','350','363 | 364','370'],xlabel='Origin 363: không đọc y[364:371] khi tạo forecast')
    save(fig,'02-time-protocol')

    fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    hist=np.array([10,12,14,16]);ax[0].plot(range(4),hist,'ko-',label='Lịch sử')
    for values,label,color in [([16]*3,'Naive',BLUE),([13]*3,'Mean4',ORANGE),([18,20,22],'Drift',GREEN)]:
        ax[0].plot([3,4,5,6],[16,*values],'o--',color=color,label=label)
    ax[0].axvline(3.5,color=GRAY,ls=':');ax[0].legend();ax[0].set(title='03 · Ví dụ tính tay bốn điểm',xlabel='Bước',ylabel='Giá trị')
    ax[1].axis('off')
    for y,txt,color in [(.83,'Lịch sử actual đến T',BLUE),(.59,'Recursive: 1 bước → nối dự báo → lặp',ORANGE),(.29,'Multi-output: 1 lần → [h1, …, h7]',GREEN)]:
        box=FancyBboxPatch((.04,y-.09),.91,.15,boxstyle='round,pad=.02',transform=ax[1].transAxes,fc=color,alpha=.12,ec=color)
        ax[1].add_patch(box);ax[1].text(.495,y-.015,txt,transform=ax[1].transAxes,ha='center',va='center',fontsize=11)
    ax[1].annotate('',xy=(.5,.68),xytext=(.5,.73),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':GRAY})
    ax[1].text(.5,.04,'Trong một rollout: không thay dự báo bằng actual tương lai',transform=ax[1].transAxes,ha='center',fontsize=10)
    save(fig,'03-baselines-strategies')

    names=list(r['validation']);labels=['Naive','Seasonal','Drift','Mean28','Ridge lag','Ridge known','Recursive','MLP known']
    fig,ax=plt.subplots(2,1,figsize=(12,8),layout='constrained');x=np.arange(8);w=.26
    for off,key,label,color in [(-w,'validation','Validation',BLUE),(0,'stable_test','Test ổn định',GREEN),(w,'test','Test shift',RED)]:
        ax[0].bar(x+off,[r[key][n]['overall']['mae'] for n in names],width=w,label=label,color=color)
    ax[0].set(xticks=x,xticklabels=labels,ylabel='MAE',title='04 · Chọn family bằng validation, không chọn lại bằng test');ax[0].legend(ncol=3)
    hist=r['mlp_history'];epochs=[v['epoch'] for v in hist]
    ax[1].plot(epochs,[v['validation_mae'] for v in hist],color=BLUE,label='Validation MAE (đơn vị target)');ax[1].axvline(r['mlp_selected_epoch'],color=ORANGE,ls='--',label='Best epoch = 131')
    ax[1].set(xlabel='Epoch fit ban đầu trên 224 ngày',ylabel='MAE',title='MLP: refit mới trên 280 ngày đúng 131 epoch');ax[1].legend()
    save(fig,'04-model-comparison')

    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    for ax,key,title in zip(axes,['validation','stable_test','test'],['Validation','Test ổn định','Test có shift']):
        for name,label,color in [('seasonal_naive','Seasonal naive',GRAY),('ridge_lags','Ridge lag',ORANGE),('ridge_known','Ridge known',BLUE),('ridge_recursive','Recursive',GREEN),('mlp_known','MLP',RED)]:
            ax.plot(range(1,8),[v['mae'] for v in r[key][name]['by_horizon']],'o-',label=label,color=color,lw=1.3,ms=3)
        ax.set(title=title,xlabel='Horizon (ngày)',ylabel='MAE',xticks=range(1,8));ax.grid(alpha=.16)
    axes[0].legend(fontsize=8);fig.suptitle('05 · Cùng target population; thêm known = cả calendar và plan',fontsize=12)
    save(fig,'05-horizon-errors')

    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained');selected=r['selected_by_validation']
    for ax,key,phase,label in [(axes[0],'stable_test','stable','Ổn định'),(axes[1],'test','shift','Có shift')]:
        result=r[key][selected];time=np.arange(364,420);y=np.array(result['actual']).ravel();pred=np.array(result['prediction']).ravel();inter=r['intervals'][phase]
        ax.fill_between(time,np.array(inter['lower']).ravel(),np.array(inter['upper']).ravel(),alpha=.22,color=BLUE,label='Interval từng điểm, nominal 80%')
        ax.plot(time,y,color='black',lw=1.4,label='Actual');ax.plot(time,pred,color=BLUE,lw=1.2,label='Forecast ghép từng tuần')
        for origin in result['origins']:ax.axvline(origin+.5,color=GRAY,ls=':',lw=.7)
        ax.set(title=f"{label} · coverage = {inter['overall']['coverage']:.1%}",ylabel='Target');ax.legend(ncol=3,fontsize=9)
    axes[1].set_xlabel('Ngày target · vạch chấm: ranh giới forecast tuần, input mới / weights cũ')
    fig.suptitle('06 · Cùng point model và q; mỗi origin phát riêng 7 bước',fontsize=12)
    save(fig,'06-rolling-intervals')

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    errors=np.sort(np.array(r['calibration']['absolute_errors']),axis=0)
    axes[0,0].plot(range(1,13),errors[:,0],'o-',color=BLUE);axes[0,0].axvline(11,color=ORANGE,ls='--');axes[0,0].axhline(r['calibration']['q'][0],color=GRAY,ls=':')
    axes[0,0].set(title='07 · Calibration h=1: chọn rank 11/12',xlabel='Thứ tự sau sort',ylabel='Absolute error',xticks=[1,4,8,11,12])
    axes[0,1].bar(range(1,8),2*np.array(r['calibration']['q']),color=BLUE);axes[0,1].set(title='Độ rộng cố định theo h',xlabel='Horizon',ylabel='Width = 2q',xticks=range(1,8))
    coverage=[r['intervals'][k]['overall']['coverage']*100 for k in ['stable','shift']];axes[1,0].bar(['Ổn định','Shift'],coverage,color=[GREEN,RED]);axes[1,0].axhline(80,color=GRAY,ls='--',label='Nominal 80%');axes[1,0].set(ylim=(0,105),ylabel='Coverage đo được (%)',title='Không suy coverage guarantee từ nominal');axes[1,0].legend()
    for i,value in enumerate(coverage):axes[1,0].text(i,value+2,f'{value:.1f}%',ha='center')
    axes[1,1].axis('off');axes[1,1].text(.04,.85,'Phát forecast + origin + version\n\n→ Actual đến: join đúng target/horizon\n\n→ Đo lỗi, bias, coverage, freshness\n\n→ Đánh giá lại fallback / recalibration / refit',va='top',transform=axes[1,1].transAxes,fontsize=11)
    save(fig,'07-calibration-monitoring')
    print(json.dumps({'png':7,'svg':7,'out':str(out)}))


if __name__=='__main__':main()
