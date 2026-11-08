"""Seven figures from executed local release report and labeled teaching diagrams."""
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
    r=json.loads(args.report.read_text());out=args.out;out.mkdir(parents=True,exist_ok=True)
    fig,ax=canvas('01 · Release pin cả bộ thành phần tương thích',
                  'Tên file model chưa mô tả đủ hành vi. Manifest liên kết object bằng digest.')
    box(ax,4.65,5.45,3.7,.9,'RELEASE MANIFEST',GREEN)
    xs=[.5,4.9,9.3]
    labels=[('MODEL + THRESHOLD','Weights / format / decision'),('PREPROCESS + CONTRACT','Feature order / scale / labels'),
            ('DATA SNAPSHOTS','Train / validation / split'),('SOURCE','Actual Python files / requirements'),
            ('ENVIRONMENT','Python / SQLite / platform'),('METRICS SNAPSHOT','Overall / slices; CI recomputes')]
    for i,(title,sub) in enumerate(labels):
        x=xs[i%3];y=3.3 if i<3 else 1.05
        box(ax,x,y,3.2,1.2,title+'\n'+sub,BLUE if i<3 else GOLD,9)
    for x in xs:arrow(ax,6.5,5.35,x+1.6,4.6)
    ax.text(6.5,.25,'Lineage được ghi rõ: weights đặt trước, không giả là model đã train từ snapshot.',ha='center',fontsize=11,color=RED)
    save(fig,'01-release-bundle',out)

    fig,ax=canvas('02 · Object bất biến, production pointer có thể đổi',
                  'Baseline và candidate chỉ đổi threshold: model/metrics đổi; các object còn lại được dùng chung.')
    box(ax,.4,4.65,3,1.1,'baseline\nthreshold = 0.6',BLUE)
    box(ax,.4,2.6,3,1.1,'candidate\nthreshold = 0.5',GREEN)
    box(ax,5,4.65,3.1,1.1,'MODEL A / METRICS A\nDigest khác B',BLUE,10)
    box(ax,5,2.6,3.1,1.1,'MODEL B / METRICS B\nDigest khác A',GREEN,10)
    arrow(ax,3.5,5.2,4.9,5.2);arrow(ax,3.5,3.15,4.9,3.15)
    box(ax,9.4,3.5,3.1,1.5,'SHARED OBJECTS\nData / source / env\nPreprocess / contract',GOLD,10)
    arrow(ax,8.2,5.2,9.3,4.6);arrow(ax,8.2,3.15,9.3,3.9)
    box(ax,2.3,.65,8.7,.85,'production: (baseline,g1) → (candidate,g2) → (baseline,g3)',GRAY,11)
    ax.text(6.5,.1,'Hash kiểm bytes so với trust root đã biết; không tự xác thực người phát hành hoặc chất lượng.',ha='center',fontsize=10,color=RED)
    save(fig,'02-content-addressing',out)

    fig,ax=canvas('03 · CI kiểm artifact, rồi controller kiểm liên kết approval',
                  'ci_runner tiêu thụ registry đã build. Không train/build lại để thay artifact sau khi PASS.')
    stages=[('BUILD ONCE','Objects + manifest'),('CONTRACT','Data / split / API'),('QUALITY','Same validation / slices'),('APPROVE','Report + policy binding')]
    for i,(title,sub) in enumerate(stages):
        x=.25+i*3.25;box(ax,x,4.4,2.7,1.2,title+'\n'+sub,BLUE if i<3 else GREEN,10)
        if i<3:arrow(ax,x+2.8,5,x+3.1,5)
    box(ax,1.4,1.7,4.25,1.3,'FAIL → giữ pointer cũ\nKhông approval cho slice-regression',RED,10)
    box(ax,7.35,1.7,4.25,1.3,'ACTIVATE → recheck\nArtifact / approval / policy / generation',GREEN,10)
    arrow(ax,6.7,4.3,3.5,3.1,RED);arrow(ax,11.1,4.3,9.5,3.1,GREEN)
    ax.text(6.5,.55,'CI PASS nghĩa đạt policy trên fixture; không tự chứng minh outcome online.',ha='center',fontsize=11)
    save(fig,'03-ci-gates',out)

    fig,ax=plt.subplots(figsize=(13,7.5));fig.subplots_adjust(left=.09,right=.97,bottom=.27,top=.75)
    names=['baseline','candidate','slice-regression'];x=np.arange(3);w=.24
    for i,(key,label,color) in enumerate([('overall','Overall, n=10',BLUE),('mobile','Mobile, n=2',GOLD),('web','Web, n=8',GREEN)]):
        values=[r['ci'][name]['observed']['overall' if key=='overall' else 'slices'] for name in names]
        heights=[v['accuracy'] if key=='overall' else v[key]['accuracy'] for v in values]
        bars=ax.bar(x+(i-1)*w,heights,w,color=color,label=label)
        for b,h in zip(bars,heights):ax.text(b.get_x()+b.get_width()/2,h+.02,f'{h:.1%}',ha='center',fontsize=9)
    ax.axhline(.75,color=RED,ls='--',lw=1,label='Min slice 75%')
    ax.set_xticks(x,['baseline\nPASS','candidate\nPASS','slice-regression\nFAIL mobile']);ax.set_ylim(0,1.26)
    ax.set_ylabel('Accuracy');ax.legend(loc='upper center',ncol=4,fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('04 · Tổng thể đạt vẫn có thể rớt gate theo slice',x=.055,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.85,'Số liệu từ report chạy thật trên validation tổng hợp; weight cố định, chỉ thay threshold.',fontsize=10,color=GRAY)
    fig.text(.055,.13,'Slice-regression overall 9/10 = 90%, nhưng mobile 1/2 = 50% < 75%.',fontsize=11,color=RED)
    fig.text(.055,.06,'N nhỏ, validation dùng để chọn threshold; không là test giữ kín hoặc bằng chứng tổng quát hóa.',fontsize=11,color=GRAY)
    save(fig,'04-quality-gates',out)

    fig,ax=canvas('05 · Rollback đổi release về trước, generation vẫn tăng',
                  'Mỗi request pin một handle. Handle đang chạy không bị thay nửa bộ weights/config.')
    for x,g,label,color in [(0.5,1,'baseline',BLUE),(4.9,2,'candidate',GREEN),(9.3,3,'baseline',BLUE)]:
        box(ax,x,4.7,3.2,1.2,f'{label} / generation {g}',color,11)
    arrow(ax,3.8,5.3,4.8,5.3);arrow(ax,8.2,5.3,9.2,5.3)
    ax.text(4.3,6,'promote',ha='center',fontsize=10);ax.text(8.7,6,'rollback',ha='center',fontsize=10)
    labels=['handle g1: baseline → priority 0','handle g2: candidate → priority 1','pin mới g3: baseline → priority 0']
    for i,text in enumerate(labels):
        y=3.25-i*.9;ax.text(.8,y,text,fontsize=12,color=GREEN if i==1 else BLUE)
        arrow(ax,7.5,y,12.25,y,GREEN if i==1 else BLUE)
    ax.text(6.5,.35,'Cùng x=[0.6,0], score≈0.598688. Threshold/version được giữ đúng theo handle.',ha='center',fontsize=11)
    save(fig,'05-generation-and-pinning',out)

    fig,ax=canvas('06 · Rollback không tự hoàn tác mọi trạng thái',
                  'Lab model read-only: chuyển pointer trong SQLite, chưa có external effects hoặc migration.')
    box(ax,.5,3.7,5.4,1.8,'BUNDLE CÓ THỂ CHỌN LẠI\nModel / threshold / preprocessing\nContract + retained dependencies',GREEN,11)
    box(ax,7.1,3.7,5.4,1.8,'CẦN XỬ LÝ RIÊNG\nEmail/tool write đã commit\nDB schema / data mất / in-flight',RED,11)
    box(ax,1.3,1.2,10.4,1.2,'Kiểm target còn load được → compatibility → switch → observed version / smoke / quality',GOLD,10)
    arrow(ax,3.2,3.6,3.2,2.5);arrow(ax,9.8,3.6,9.8,2.5)
    ax.text(6.5,.35,'Có thể cần compensation, forward fix hoặc fallback có quyền; old weights chưa đủ.',ha='center',fontsize=11)
    save(fig,'06-rollback-boundaries',out)

    fig,ax=canvas('07 · Quy trình release có đường quay lại và kiểm sau chuyển',
                  'Sơ đồ học tập; demo chỉ chạy local. Không có remote deployment hoặc canary routing thật.')
    top=[('PIN INPUTS','Source / data / config'),('CI + RETAIN','Bundle / report / target'),('STAGED RELEASE','Observe route / slice')]
    bottom=[('INVESTIGATE','Keep journal / artifacts'),('VERIFY AFTER','Observed version / smoke'),('SWITCH TARGET','CAS + compatible bundle')]
    for i,(a,b) in enumerate(top):box(ax,.5+4.4*i,4.5,3.2,1.2,a+'\n'+b,BLUE,10)
    for i,(a,b) in enumerate(bottom):box(ax,.5+4.4*i,1.65,3.2,1.2,a+'\n'+b,GREEN,10)
    arrow(ax,3.8,5.1,4.8,5.1);arrow(ax,8.2,5.1,9.2,5.1)
    arrow(ax,10.9,4.4,10.9,2.95,RED);ax.text(10.9,3.65,'incident / stop expansion',ha='center',fontsize=9,color=RED,bbox={'facecolor':'white','edgecolor':'none','pad':3})
    arrow(ax,9.2,2.25,8.2,2.25);arrow(ax,4.8,2.25,3.8,2.25)
    ax.text(6.5,.45,f"Lab: {r['evaluation']['passed']}/{r['evaluation']['checks']} regression checks; fault rollback, restart và hai controller cục bộ.",ha='center',fontsize=11)
    save(fig,'07-release-runbook',out)


if __name__=='__main__':main()
