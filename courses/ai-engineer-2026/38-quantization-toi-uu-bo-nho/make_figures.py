"""Original publication figures from equations and an actual CPU experiment report."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).parent/'figures');a=p.parse_args()
    r=json.loads(a.report.read_text());e=r['experiment'];rows={row['name']:row for row in e['rows']};a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    blue,green,orange,red='#245ca6','#25836a','#dd8a21','#be4b53'
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(a.out/f'{name}.{ext}',dpi=155,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    def box(ax,x,y,text,w=.25,h=.21,color=blue):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=color+'14',ec=color,lw=1.5))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
    def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':'#566577','lw':1.6})
    fig,ax=plt.subplots(figsize=(12,5.4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('4-bit storage chưa mô tả toàn bộ inference memory',fontsize=17,pad=18)
    box(ax,.03,.62,'Persistent weights\ncodes + scales + bias',color=blue)
    box(ax,.38,.62,'Dequant tạm\nŴ FP32 trong lab',color=orange)
    box(ax,.73,.62,'Matmul trong lab\nFP32 input / compute',w=.24,color=green)
    arrow(ax,.28,.72,.38,.72);arrow(ax,.63,.72,.73,.72)
    box(ax,.03,.21,'Activations\nfake-quant vẫn FP32',color=red)
    box(ax,.38,.21,'KV cache\nđối tượng quant riêng',color=blue)
    box(ax,.73,.21,'Peak process/device\n+ copies / workspace',w=.24,color=orange)
    ax.text(.5,.05,'Đếm tensor state, file size và đo peak memory là ba việc khác nhau.',ha='center',fontsize=12)
    save(fig,'01-memory-map')

    x=np.linspace(-5,5,1200);fig,axs=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    axs[0].plot(x,x,color='gray',ls='--',label='Giá trị gốc');axs[0].plot(x,np.clip(np.rint(x/.5),-7,7)*.5,color=blue,label='Narrow INT4, s=.5')
    axs[0].set(xlabel='x',ylabel='x̂',title='Lưới đều + saturation tại ±3.5');axs[0].legend(fontsize=9)
    axs[1].plot(x,x,color='gray',ls='--',label='Giá trị gốc');axs[1].plot(x,np.clip(np.rint(x*127),-127,127)/127,color=red,label='Static INT8, range ±1')
    axs[1].axvspan(-5,-1,color=red,alpha=.08);axs[1].axvspan(1,5,color=red,alpha=.08)
    axs[1].set(xlabel='x khi dữ liệu đổi range',ylabel='x̂',title='x=4 → x̂=1: clipping không chỉ rounding');axs[1].legend(fontsize=9)
    fig.suptitle('Scale nhỏ làm lưới mịn, range hẹp có thể clip nhiều hơn',fontsize=15)
    save(fig,'02-grid-clipping')

    fig,axs=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained');xx=np.arange(3)
    w=np.array([.1,.2,.3]);per_row=np.rint(w/(.3/7))*(.3/7)
    for shift,values,c,label in [(-.24,w,blue,'Gốc'),(0,np.zeros(3),red,'Scale chung 30/7'),(.24,per_row,green,'Scale riêng .3/7')]:axs[0].bar(xx+shift,values,.24,color=c,label=label)
    axs[0].set_xticks(xx,['w₀','w₁','w₂']);axs[0].set(ylabel='Giá trị hàng nhỏ',ylim=(0,.41));axs[0].legend(fontsize=9);axs[0].set_title('Outlier ở hàng khác làm mất hàng nhỏ',fontsize=11)
    labels=['W4-row','W4-g8','W4-g32'];code=np.array([408,408,816]);scale=np.array([204,408,204]);other=np.array([268]*3)
    axs[1].bar(xx,code,color=blue,label='Codes');axs[1].bar(xx,scale,bottom=code,color=orange,label='Scales FP32');axs[1].bar(xx,other,bottom=code+scale,color=green,label='Bias + mean/std')
    axs[1].set_xticks(xx,labels);axs[1].set(ylabel='Persistent tensor byte',ylim=(0,1650));axs[1].legend(fontsize=9,ncol=1)
    for i,label in enumerate(labels):assert rows[label]['persistent_tensor_bytes']==code[i]+scale[i]+other[i]
    fig.suptitle('Granularity đổi sai số; group/padding đổi tổng số byte',fontsize=15)
    save(fig,'03-packing-granularity')

    names=['FP32','W4-g8-A8static','W4-g8-A8dynamic'];xx=np.arange(3);fig,axs=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    for offset,key,c,label in [(-.18,'test',blue,'Test'),(.18,'stress4x',orange,'Stress ×4')]:
        values=[rows[n][key]['correct'] for n in names];axs[0].bar(xx+offset,values,.36,color=c,label=label)
        for i,v in enumerate(values):axs[0].text(i+offset,v+1,str(v),ha='center',fontsize=10)
    axs[0].set_xticks(xx,['FP32','Static A8','Dynamic A8']);axs[0].set(ylabel='Số đúng / 132 row',ylim=(0,148));axs[0].legend(fontsize=9,loc='lower left')
    static=rows['W4-g8-A8static'];clipped=[100*static[key]['activation_clipped']/static[key]['activation_elements'] for key in ['test','stress4x']]
    axs[1].bar(['Test','Stress ×4'],clipped,color=[blue,red]);axs[1].set(ylabel='% input Linear vượt static range',ylim=(0,10.5))
    for i,v in enumerate(clipped):axs[1].text(i,v+.2,f'{v:.3f}%',ha='center')
    fig.suptitle('Calibration chỉ train; stress ×4 là biến đổi từ cùng test',fontsize=15)
    save(fig,'04-calibration-shift')

    m=r['hand']['memory'];weights=np.array([m['fp16_weight_bytes'],m['mixed_weight_bytes']])/1024**3;other=np.array([2.,2.]);kv=np.array([4.,4.]);xx=np.arange(2)
    fig,ax=plt.subplots(figsize=(10.5,5.3),layout='constrained')
    ax.bar(xx,weights,color=blue,label='Weights');ax.bar(xx,other,bottom=weights,color=orange,label='Other: giả định 2GiB');ax.bar(xx,kv,bottom=weights+other,color=green,label='KV: 8 × .5GiB')
    ax.axhline(16,color=red,ls='--',label='Budget16GiB');ax.set_xticks(xx,['Weights FP16','Mixed W4 + scales + FP16']);ax.set(ylabel='GiB (ledger giả định)',ylim=(0,23.5))
    for i,v in enumerate(weights+other+kv):ax.text(i,v+.3,f'{v:.3f}GiB',ha='center')
    ax.legend(loc='upper right',fontsize=10);ax.set_title('7B giả định: KV/workspace chưa biến mất khi giảm weights',fontsize=15,pad=18)
    save(fig,'05-memory-ledger')

    labels=['FP32','W8\nrow','W4\nrow','W4\ng8','W4\ng32','W4-g8\nA8static','W4-g8\nA8dynamic'];values=list(rows.values());xx=np.arange(7);fig,axs=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained')
    colors=[blue,green,orange,green,orange,red,green]
    axs[0].bar(xx,[v['persistent_tensor_bytes'] for v in values],color=colors);axs[0].set(ylabel='Persistent tensor byte');axs[0].set_xticks(xx,labels,fontsize=9)
    axs[1].bar(xx,[v['test']['logit_rmse'] for v in values],color=colors);axs[1].set(ylabel='Logit RMSE test so FP32');axs[1].set_xticks(xx,labels,fontsize=9)
    fig.suptitle('MLP867parameter: byte tiết kiệm và lỗi đầu ra là hai phép đo',fontsize=15)
    save(fig,'06-quality-memory')

    kvrows=r['attention']['rows'];labels=['FP32','INT8','Packed INT4'];xx=np.arange(3);fig,axs=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    payload=[r['attention']['fp32_kv_bytes']]+[v['packed_payload_bytes'] for v in kvrows];err=[0]+[v['rmse'] for v in kvrows]
    axs[0].bar(xx,payload,color=[blue,green,orange]);axs[0].set_xticks(xx,labels);axs[0].set(ylabel='K/V code + scale byte',ylim=(0,5000))
    for i,v in enumerate(payload):axs[0].text(i,v+80,str(v),ha='center')
    axs[1].bar(xx,err,color=[blue,green,orange]);axs[1].set_xticks(xx,labels);axs[1].set(ylabel='Attention output RMSE',ylim=(0,.043))
    fig.suptitle('Một attention diagnostic: K/V[2,2,17,8], group8; chưa cache engine',fontsize=14)
    save(fig,'07-kv-attention')
    print(json.dumps({'figures':7,'formats':['png','svg'],'out':str(a.out)},ensure_ascii=False))


if __name__=='__main__':main()
