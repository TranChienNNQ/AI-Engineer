"""Original figures; learning curves regenerated with the actual lab in a temporary run."""
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import contextlib
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lab import study

OUT=Path(__file__).resolve().parent/'figures'
BLUE='#2563eb'; GREEN='#15803d'; ORANGE='#ea580c'

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150)
    fig.savefig(OUT/(name+'.svg'))
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,ax=plt.subplots(figsize=(11,4.2),layout='constrained');ax.axis('off')
    labels=['Train\nforward → backward → step','Validation\nmodel cố định, chỉ đo','Chọn best\nkiểm patience','Test sau lựa chọn\nkhông dùng để chọn lại']
    positions=[(.22,.72),(.76,.72),(.76,.25),(.22,.25)]
    for (x,y),t,c in zip(positions,labels,['#dbeafe','#dcfce7','#ffedd5','#f3e8ff']):
        ax.text(x,y,t,ha='center',va='center',bbox={'boxstyle':'round,pad=.8','facecolor':c})
    for start,end in [((.43,.72),(.57,.72)),((.76,.59),(.76,.40)),((.56,.25),(.43,.25))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':2})
    ax.plot([.64,.50,.22],[.37,.50,.50],color=BLUE)
    ax.annotate('',xy=(.22,.59),xytext=(.22,.50),arrowprops={'arrowstyle':'->','color':BLUE})
    ax.text(.43,.54,'Epoch kế tiếp nếu chưa dừng',ha='center',color=BLUE,fontsize=10)
    ax.text(.50,.16,'Sau khi dừng và chốt run',ha='center',fontsize=10)
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.set_title('Một run: train cập nhật, validation lựa chọn; test sau khi chốt',pad=15)
    save(fig,'01-lifecycle')
    with TemporaryDirectory(prefix='lesson12-figures-') as directory:
        with contextlib.redirect_stdout(io.StringIO()):report=study(Path(directory)/'study')
        histories={r['run']:list(csv.DictReader(open(Path(directory)/'study'/r['run']/'metrics.csv'))) for r in report['candidates']}
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
    rows=histories['lr-003'];epochs=[int(r['epoch']) for r in rows]
    for col,label,color in [('train_online_mse','Train trong lúc update',ORANGE),('train_eval_mse','Train đo lại (eval)',BLUE),('val_mse','Validation (eval)',GREEN)]:
        ax.plot(epochs,[float(r[col]) for r in rows],label=label,color=color,alpha=.85)
    ax.axvline(58,color='gray',ls='--',label='Best val: epoch 58')
    ax.set(xlabel='Epoch',ylabel='Mean squared error',title='Run lr=0.03: ba phép đo khác nhau')
    ax.legend(fontsize=10);ax.grid(alpha=.2);save(fig,'02-learning-curves')
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.plot([1,2,3,4],[.8,.6,.595,.61],'-o',color=BLUE,label='Đã quan sát')
    ax.scatter([5],[.58],marker='x',s=90,color='gray',label='Giả định, chưa chạy')
    ax.axvline(4,color=ORANGE,ls='--',label='Dừng sau epoch 4')
    ax.scatter([2],[.6],s=160,facecolors='none',edgecolors=GREEN,lw=2,label='Giữ epoch 2')
    ax.set(xticks=[1,2,3,4,5],xlabel='Epoch',ylabel='Validation MSE',title='Ví dụ tính tay: patience=2, min_delta=0.010')
    ax.legend(fontsize=10);ax.grid(alpha=.2);save(fig,'03-selection')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,title,body,color in zip(axes,['best.pt: dự đoán model đã chọn','last.pt: tiếp tục tiến trình'],[
        'Model state ở best epoch\nConfig dựng kiến trúc\nBest epoch + validation MSE\nData hash + schema\n\nLoad → eval → no_grad',
        'Model + optimizer state hiện tại\nEpoch + global_step\nRNG CPU + bộ đếm patience\nBest snapshot + history + config\n\nLoad → epoch+1 → train'],['#dcfce7','#dbeafe']):
        ax.axis('off');ax.set_title(title,pad=15)
        ax.text(.5,.5,body,ha='center',va='center',linespacing=1.8,bbox={'boxstyle':'round,pad=1','facecolor':color})
    fig.suptitle('Hai artifact phục vụ hai mục đích; không ghép trạng thái khác epoch')
    save(fig,'04-checkpoints')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for r,color in zip(report['candidates'],[BLUE,ORANGE]):
        rows=histories[r['run']]
        axes[0].plot([int(x['epoch']) for x in rows],[float(x['val_mse']) for x in rows],label=f"lr={r['lr']}",color=color)
        axes[0].scatter([r['best_epoch']],[r['best_val']],color=color,s=70,zorder=3)
    axes[0].set(title='Chỉ dùng validation để so sánh',xlabel='Epoch',ylabel='Validation MSE');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].axis('off');axes[1].text(.5,.55,'Chọn lr=0.03, epoch 58\nVal MSE ≈ 0.194300\n↓\nChấm test một lần: 21 mẫu\nTest MSE ≈ 0.215744',ha='center',va='center',linespacing=1.8,bbox={'boxstyle':'round,pad=.8','facecolor':'#f3e8ff'})
    axes[1].set_title('Báo cáo sau lựa chọn')
    save(fig,'05-tracking');print('Created 5 PNG + 5 SVG')

if __name__=='__main__':main()
