"""Generate five statistical figures (PNG + SVG), using the lesson's lab data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

from lab import SEED, calibration_example, descriptive_stats, sample_proportions, wilson_interval


OUT=Path(__file__).resolve().parent/"figures"
BLUE,ORANGE,GREEN,RED="#2563eb","#d97706","#15803d","#dc2626"


def save(fig,name):
    fig.savefig(OUT/f"{name}.png",dpi=170)
    fig.savefig(OUT/f"{name}.svg")
    plt.close(fig)


def statistics_figure():
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.0),layout="constrained")
    for ax,values,title in zip(axes,([20]*5,[10]*4+[60]),("Cụm ổn định","Có đuôi lớn")):
        s=descriptive_stats(values)
        ax.hist(values,bins=np.arange(5,76,10),color=BLUE,alpha=.65,edgecolor="white")
        ax.axvline(s["mean"],color=ORANGE,lw=2,label=f"Mean = {s['mean']:g} ms")
        ax.axvline(s["median"],color=GREEN,lw=2,ls="--",label=f"Median = {s['median']:g} ms")
        ax.set(title=title,xlabel="Latency giả định (ms)",ylabel="Số quan sát",xlim=(0,70),ylim=(0,5.8))
        ax.set_yticks(range(6));ax.legend(loc="upper right",fontsize=10)
        ax.text(.97,.63,f"n = 5\nVar(ddof=0) = {s['var_ddof0']:g} ms²\nSD(ddof=0) = {s['std_ddof0']:g} ms",
                ha="right",va="top",transform=ax.transAxes,fontsize=10)
    fig.suptitle("Cùng mean 20 ms, khác phân bố và mức phân tán",fontsize=16,fontweight="bold")
    save(fig,"01-thong-ke")


def bayes_figure():
    fig,ax=plt.subplots(figsize=(12.5,6.2),layout="constrained")
    ax.axis("off");ax.set(xlim=(0,1),ylim=(-.07,1.04))
    def node(x,y,text,color,width=.19):
        patch=FancyBboxPatch((x-width/2,y-.055),width,.11,
                            boxstyle="round,pad=0.01",facecolor=color,edgecolor="#94a3b8")
        ax.add_patch(patch);ax.text(x,y,text,ha="center",va="center",fontsize=11)
    def edge(start,end,label,pos):
        ax.annotate("",xy=end,xytext=start,arrowprops={"arrowstyle":"->","color":"#64748b","lw":1.6})
        ax.text(*pos,label,ha="center",fontsize=10,color="#334155")
    node(.11,.5,"10 000 requests","#f1f5f9",.18)
    node(.43,.77,"Có lỗi: 100","#dbeafe")
    node(.43,.23,"Không lỗi: 9 900","#f1f5f9")
    node(.85,.91,"Báo đúng: 90","#dcfce7",.25)
    node(.85,.64,"Bỏ sót: 10","#fef3c7",.25)
    node(.85,.37,"Báo nhầm: 495","#fee2e2",.25)
    node(.85,.10,"Không báo: 9 405","#f1f5f9",.25)
    edge((.21,.52),(.325,.76),"1%",(.27,.71))
    edge((.21,.48),(.325,.24),"99%",(.27,.31))
    edge((.535,.79),(.715,.90),"90%",(.63,.90))
    edge((.535,.75),(.715,.65),"10%",(.63,.64))
    edge((.535,.25),(.715,.36),"5%",(.63,.36))
    edge((.535,.21),(.715,.11),"95%",(.63,.10))
    ax.set_title("Bayes bằng bảng đếm: lỗi hiếm, báo nhầm vẫn chiếm đa số cảnh báo",fontsize=15,fontweight="bold",pad=18)
    ax.text(.5,-.045,"Trong 585 cảnh báo: 90 có lỗi → P(lỗi | cảnh báo) = 90/585 ≈ 15.38%",
            ha="center",fontsize=12,fontweight="bold",color=BLUE)
    save(fig,"02-bayes")


def sampling_figure():
    fig,axes=plt.subplots(1,2,figsize=(12.0,5.0),layout="constrained",sharex=True)
    for ax,n in zip(axes,(100,1000)):
        estimates=sample_proportions(.2,n,5000,SEED)
        ax.hist(estimates,bins=np.arange(-.005,1.006,.01),color=BLUE,alpha=.75,edgecolor="white")
        ax.axvline(.2,color=ORANGE,lw=2,label="p thật = 0.2")
        ax.set(title=f"Mỗi thí nghiệm n = {n}",xlabel="Tỷ lệ quan sát p̂",ylabel="Số thí nghiệm",xlim=(0,.4))
        ax.legend();ax.text(.97,.75,f"SE lý thuyết = {np.sqrt(.2*.8/n):.4f}",
                            transform=ax.transAxes,ha="right",fontsize=10)
    fig.suptitle("Lặp 5 000 thí nghiệm độc lập: tỷ lệ ước lượng cũng có phân bố",fontsize=15,fontweight="bold")
    save(fig,"03-lay-mau")


def intervals_figure():
    fig,axes=plt.subplots(1,2,figsize=(12.5,6.2),layout="constrained")
    counts=np.random.default_rng(SEED).binomial(100,.2,size=50)
    covered=[]
    for row,k in enumerate(counts):
        lo,hi=wilson_interval(int(k),100)
        ok=lo<=.2<=hi;covered.append(ok)
        color=GREEN if ok else RED
        axes[0].plot([lo,hi],[row,row],color=color,lw=1.7)
        axes[0].scatter(k/100,row,color=color,s=9)
    axes[0].axvline(.2,color="#334155",ls="--",lw=1.4)
    axes[0].set(title=f"50 khoảng từ 50 mẫu: {sum(covered)} chứa p thật",
                xlabel="p và khoảng Wilson",ylabel="Thí nghiệm độc lập (n=100)",xlim=(0,.45))
    axes[0].invert_yaxis();axes[0].legend(handles=[Patch(color=GREEN,label="Chứa p=0.2"),Patch(color=RED,label="Không chứa")],fontsize=9)
    for row,(k,n,color) in enumerate([(20,100,BLUE),(200,1000,ORANGE)]):
        lo,hi=wilson_interval(k,n);p=k/n
        axes[1].errorbar(p,row,xerr=[[p-lo],[hi-p]],fmt="o",capsize=6,color=color,lw=2)
    axes[1].set(yticks=[0,1],yticklabels=["20/100","200/1 000"],ylim=(-.7,1.7),
                xlim=(.1,.32),xlabel="p̂ và khoảng Wilson",title="Cùng tỷ lệ 20%, khác độ bất định")
    axes[1].axvline(.2,color="#334155",ls="--",lw=1)
    fig.suptitle("Khoảng tin cậy danh nghĩa 95% mô tả một quy trình lấy mẫu",fontsize=15,fontweight="bold")
    save(fig,"04-khoang-tin-cay")


def calibration_figure():
    y,p_a,p_b=calibration_example()
    observed=[float(np.mean(y[i:i+10])) for i in range(0,30,10)]
    fig,ax=plt.subplots(figsize=(8.0,6.3),layout="constrained")
    ax.plot([0,1],[0,1],"--",color="#64748b",label="Đường tham chiếu y=x")
    for name,p,color in [("A",p_a,BLUE),("B",p_b,ORANGE)]:
        points=[float(np.mean(p[i:i+10])) for i in range(0,30,10)]
        ax.plot(points,observed,"o-",color=color,lw=2,label=f"Mô hình {name}")
    ax.set(title="Cùng accuracy 70%, chất lượng xác suất khác nhau",
           xlabel="Xác suất dự đoán của nhóm",ylabel="Tỷ lệ nhãn 1 quan sát trong nhóm",
           xlim=(-.03,1.03),ylim=(-.03,1.03),aspect="equal")
    ax.legend(loc="upper left");ax.grid(alpha=.2)
    ax.text(.97,.04,"Dữ liệu tự tạo: 10 mẫu/điểm\nĐoạn nối chỉ dẫn mắt, không phải kết quả nội suy",
            transform=ax.transAxes,ha="right",fontsize=9,color="#475569")
    save(fig,"05-calibration")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "figure.facecolor":"white","savefig.facecolor":"white"})
    statistics_figure();bayes_figure();sampling_figure();intervals_figure();calibration_figure()
    print(f"Generated 5 PNG and 5 SVG figures in {OUT}")


if __name__=="__main__":
    main()
