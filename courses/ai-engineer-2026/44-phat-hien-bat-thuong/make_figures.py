"""Seven static PNG/SVG pairs; measured comparisons come from a full lab report."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import tabular,temporal,calendar
from detectors import fit,score
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':12,'figure.dpi':130,'savefig.dpi':150,'svg.hashsalt':'lesson44'})
BLUE='#2463a6';ORANGE='#dc7c30';GREEN='#26836d';RED='#bd4359';GRAY='#666f7c'


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).parent/'figures');a=p.parse_args()
    r=json.loads(a.report.read_text())['experiment'];a.out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for suffix in ['png','svg']:
            metadata={'Software':'lesson44'} if suffix=='png' else {'Date':None,'Creator':'lesson44'}
            fig.savefig(a.out/(name+'.'+suffix),bbox_inches='tight',facecolor='white',metadata=metadata)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    axes[0].scatter(range(5),[10,11,12,13,14],color=BLUE,label='Reference');axes[0].scatter([5],[20],color=RED,s=65,label='Query mới')
    axes[0].axhline(12,color=GRAY,ls=':');axes[0].set(title='Điểm riêng lẻ',ylabel='Giá trị');axes[0].legend(fontsize=9)
    t=np.arange(48);expected=50+20*np.sin(2*np.pi*t/24);actual=expected.copy();actual[18]=50
    axes[1].plot(t,expected,color=BLUE,label='Kỳ vọng theo giờ');axes[1].scatter([18],[actual[18]],color=RED,s=55,label='Lạ theo context')
    axes[1].set(title='Không quá lớn, vẫn lạ',xlabel='Giờ',ylabel='Giá trị');axes[1].legend(fontsize=9)
    tt=np.arange(24);values=np.sin(tt)*.15;values[9:17]+=1
    axes[2].plot(tt,values,'o-',ms=3,color=BLUE);axes[2].axvspan(8.5,16.5,color=RED,alpha=.12);axes[2].set(title='Đoạn lệch kéo dài',xlabel='Bước',ylabel='Residual minh họa')
    fig.suptitle('01 · Minh họa khái niệm: cần định nghĩa reference và context',fontsize=12)
    save(fig,'01-anomaly-context')

    d=tabular();x=d['x'][:300];mu=x.mean(0);std=x.std(0);z=(x-mu)/std;query=(d['x'][620:]-mu)/std
    gx,gy=np.meshgrid(np.linspace(-4.5,4.5,91),np.linspace(-4.5,4.5,91));grid=np.column_stack([gx.ravel(),gy.ravel()]);fig,axes=plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for ax,kind,label in zip(axes.flat,['zmax','knn','pca_residual','mahalanobis'],['Max z: không xét tương quan','kNN: xa reference','PCA: khoảng cách ngoài đường','Mahalanobis: covariance distance']):
        m=fit(x,kind);q=r['tabular']['detectors'][kind]['threshold'];surface=score(m,grid*std+mu).reshape(gx.shape)
        ax.contourf(gx,gy,surface>q,levels=[-.5,.5,1.5],colors=['#e8f2f9','#f9e8ec'],alpha=.7)
        ax.contour(gx,gy,surface,levels=[q],colors=GRAY,linewidths=1)
        ax.scatter(z[:,0],z[:,1],s=5,alpha=.25,color=BLUE,label='Train normal')
        for kind_,marker,color in [('off','x',RED),('along','^',ORANGE)]:
            ix=d['kind'][620:]==kind_;ax.scatter(query[ix,0],query[ix,1],marker=marker,s=42,color=color,label=kind_)
        ax.set(title=label,xlabel='Feature 1 chuẩn hóa train',ylabel='Feature 2 chuẩn hóa train',xlim=(-4.5,4.5),ylim=(-4.5,4.5));ax.set_aspect('equal')
    axes[0,0].legend(fontsize=8,loc='upper left');fig.suptitle('02 · Vùng hồng: score > ngưỡng calibration; điểm test chỉ để minh họa',fontsize=11)
    save(fig,'02-multivariate-geometry')

    fig,axes=plt.subplots(2,1,figsize=(12,6.2),layout='constrained')
    stages=[(0,300,'Fit reference',BLUE),(300,200,'Calibration q',GREEN),(500,120,'Validation F1',ORANGE),(620,180,'Test',RED)]
    for start,width,label,color in stages:
        axes[0].broken_barh([(start,width)],(.7,.6),facecolors=color);axes[0].text(start+width/2,1,label,ha='center',va='center',color='white')
    axes[0].set(xlim=(-5,805),ylim=(.1,1.8),yticks=[],xticks=[0,300,500,620,800],title='03 · Bảng: fit → calibrate từng score → chọn family → test',xlabel='Index, boundary phải exclusive · không refit sau selection')
    axes[0].text(250,.3,'300 train + 200 calibration: giả định normal',ha='center',fontsize=10)
    for start,label,color in [(0,'Fit calendar/scales',BLUE),(168,'Calibration q',GREEN),(336,'Test + policy',RED)]:
        axes[1].broken_barh([(start,168)],(.7,.6),facecolors=color);axes[1].text(start+84,1,label,ha='center',va='center',color='white')
    axes[1].set(xlim=(-5,509),ylim=(.1,1.8),yticks=[],xticks=[0,168,336,504],title='Chuỗi: cấu hình cố định; CUSUM state liên tục từ 168',xlabel='Index giờ · y hiện tại được đọc khi đã quan sát')
    axes[1].text(252,.3,'Không dùng label để cập nhật baseline, reset hoặc suppress score',ha='center',fontsize=10)
    save(fig,'03-data-protocol')

    families=list(r['tabular']['detectors']);names=['zmax','MAD max','kNN','PCA residual','Mahalanobis²'];vals=r['tabular']['detectors'];ix=np.arange(5);w=.35
    fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
    axes[0].bar(ix-w/2,[vals[k]['validation']['point']['f1'] for k in families],w,color=BLUE,label='Validation F1');axes[0].bar(ix+w/2,[vals[k]['test']['point']['f1'] for k in families],w,color=GREEN,label='Test F1')
    axes[0].set(xticks=ix,xticklabels=names,ylim=(0,1.12),ylabel='F1',title='04 · Chọn kNN bằng validation; không chọn lại bằng test');axes[0].legend(ncol=2)
    for off,group,color in [(-w/2,'off',RED),(w/2,'along',ORANGE)]:axes[1].bar(ix+off,[vals[k]['test']['recall_by_kind'][group] for k in families],w,color=color,label=f'Recall {group}')
    axes[1].set(xticks=ix,xticklabels=names,ylim=(0,1.12),ylabel='Recall',title='Test có 9 off + 9 along; PCA residual có điểm mù');axes[1].legend(ncol=2)
    save(fig,'04-tabular-results')

    time=temporal();t=np.arange(336,504);actual=time['y'][336:];expect=calendar(t,time['plan'][336:])@np.array(r['temporal']['fit']['coef']);truth=[(354,355),(384,392),(426,432),(468,484)]
    def mark(ax):
        for start,end in truth:ax.axvspan(start-.5,end-.5,color=RED,alpha=.12)
        ax.axvspan(359.5,365.5,color=BLUE,alpha=.10)
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    axes[0].plot(t,actual,color='black',lw=1.2,label='Actual');axes[0].plot(t,expect,color=BLUE,lw=1.3,label='Calendar expectation đã fit');mark(axes[0])
    axes[0].set(title='05 · Test theo giờ: vùng đỏ = anomaly; vùng xanh = plan normal',ylabel='Giá trị target');axes[0].legend(ncol=2)
    det=r['temporal']['detectors']['calendar_residual'];axes[1].plot(t,det['test']['scores'],color=GREEN,label='Absolute standardized residual');axes[1].axhline(det['threshold'],color=GRAY,ls='--',label='Ngưỡng cố định');mark(axes[1])
    axes[1].set(xlabel='Index giờ (0-based)',ylabel='Score');axes[1].legend(ncol=2)
    save(fig,'05-temporal-observations')

    fig,axes=plt.subplots(4,1,figsize=(12,10),layout='constrained')
    for ax,(name,v) in zip(axes,r['temporal']['detectors'].items()):
        sc=np.array(v['test']['scores'])/v['threshold'];ax.plot(t,sc,color=BLUE,lw=1.2);ax.axhline(1,color=GRAY,ls='--',lw=1);mark(ax)
        fires=v['policy']['firings']
        if fires:ax.scatter(fires,[sc[f-336] for f in fires],color=ORANGE,marker='v',s=60,zorder=3,label='New firing')
        ax.set(title=f"{name} · {len(fires)} firing · policy tìm {v['alert_events']['detected']}/4 events",ylabel='Score / q')
        if fires:ax.legend(loc='upper left',fontsize=8)
    axes[-1].set_xlabel('Index giờ · policy 2 hits / 3 non-hits / cooldown 4 · không reset CUSUM')
    fig.suptitle('06 · Score vượt ngưỡng khác với mở một cảnh báo mới',fontsize=12)
    save(fig,'06-scores-alerts')

    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    bars=axes[0].bar(['True positives','False positives'],[90,999],color=[GREEN,RED]);axes[0].bar_label(bars,padding=3)
    axes[0].set(ylim=(0,1300),ylabel='Số lượng kỳ vọng / 100.000 points',title='07 · Prevalence 0.1%, TPR 90%, FPR 1%')
    axes[0].text(.5,.91,'Precision = 90 / 1089 ≈ 8.26%',ha='center',transform=axes[0].transAxes,fontsize=11)
    illustration=np.array([[1]*10,[0]*9+[1],[1]*10]);axes[1].imshow(illustration,cmap=matplotlib.colors.ListedColormap(['#eff2f5',RED]),vmin=0,vmax=1,aspect='auto')
    axes[1].set(yticks=[0,1,2],yticklabels=['Truth event','Raw prediction','Sau point adjustment'],xticks=range(10),xlabel='Bước từ onset: raw delay = 9',title='Một hit cuối đoạn không phải phát hiện sớm')
    axes[1].text(.5,-.28,'PA chỉ minh họa tác động; không áp dụng trong lab',ha='center',transform=axes[1].transAxes,fontsize=9)
    save(fig,'07-evaluation-pitfalls')
    print(json.dumps(dict(png=7,svg=7,out=str(a.out))))


if __name__=='__main__':main()
