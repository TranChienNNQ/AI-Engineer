"""Create five lesson diagrams from the same partitions and examples as lab.py."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

from lab import (SEED, choose_k, cv_folds, grouped_data, group_partitions,
                 temporal_partitions, training_data, validation_data)


OUT=Path(__file__).resolve().parent/"figures"
BLUE,ORANGE,GREEN,RED,GRAY="#2563eb","#d97706","#15803d","#dc2626","#cbd5e1"
CMAP=ListedColormap([BLUE,ORANGE,GREEN,RED,GRAY])
NORM=BoundaryNorm(np.arange(-.5,5.5),CMAP.N)


def save(fig,name):
    fig.savefig(OUT/f"{name}.png",dpi=170)
    fig.savefig(OUT/f"{name}.svg")
    plt.close(fig)


def roles_figure():
    fig,ax=plt.subplots(figsize=(12.5,4.8),layout="constrained")
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    boxes=[(.03,"TRAIN","Fit mô hình\nvà thống kê tiền xử lý",BLUE,"#dbeafe"),
           (.375,"VALIDATION","Chọn cấu hình, ngưỡng\nvà thời điểm dừng",ORANGE,"#ffedd5"),
           (.72,"TEST","Đánh giá quy trình\nđã đóng băng",GREEN,"#dcfce7")]
    for x,title,body,color,bg in boxes:
        ax.add_patch(FancyBboxPatch((x,.42),.25,.34,boxstyle="round,pad=.012",facecolor=bg,edgecolor=color,lw=1.5))
        ax.text(x+.125,.67,title,color=color,ha="center",fontweight="bold",fontsize=14)
        ax.text(x+.125,.53,body,ha="center",va="center",fontsize=11)
    for start,end,label in [(.29,.36,"Đo trên val"),(.635,.705,"Chốt lựa chọn")]:
        ax.annotate("",xy=(end,.59),xytext=(start,.59),arrowprops={"arrowstyle":"->","lw":1.8,"color":"#475569"})
        ax.text((start+end)/2,.82,label,ha="center",fontsize=10)
    ax.annotate("",xy=(.155,.40),xytext=(.5,.40),
                arrowprops={"arrowstyle":"->","connectionstyle":"arc3,rad=-.45","color":ORANGE,"lw":1.6})
    ax.text(.33,.12,"Đổi cấu hình → fit lại trên train",ha="center",color=ORANGE,fontsize=11)
    ax.text(.5,.04,"Dùng kết quả test để sửa lựa chọn → test ấy không còn độc lập với quá trình phát triển",
            ha="center",fontsize=11,color="#475569")
    ax.set_title("Ba vai trò dữ liệu, không chỉ ba tên thư mục",fontsize=17,fontweight="bold",pad=15)
    save(fig,"01-vai-tro")


def selection_figure():
    x,y=training_data();vx,vy=validation_data()
    selected,rows=choose_k(x,y,vx,vy)
    fig,axes=plt.subplots(1,2,figsize=(12.3,5.1),layout="constrained")
    axes[0].scatter(x,y,color=BLUE,s=55,label="Train: 6 mẫu")
    axes[0].scatter(vx,vy,marker="s",facecolors="none",edgecolors=ORANGE,s=65,label="Validation: 3 mẫu")
    axes[0].set(title="Dữ liệu dùng trước khi mở test",xlabel="Đặc trưng tải x (đơn vị quy ước)",ylabel="Đích giả định y (ms)")
    axes[0].legend();axes[0].grid(alpha=.18)
    ks=[r[0] for r in rows]
    axes[1].plot(ks,[r[1] for r in rows],"o-",color=BLUE,label="Train MSE")
    axes[1].plot(ks,[r[2] for r in rows],"o-",color=ORANGE,label="Validation MSE")
    best=next(r for r in rows if r[0]==selected)
    axes[1].scatter(selected,best[2],marker="*",color=GREEN,s=170,zorder=5)
    axes[1].annotate("Chọn k=3 bằng validation",(selected,best[2]),xytext=(1.45,.65),
                     arrowprops={"arrowstyle":"->","color":GREEN},color=GREEN,fontsize=10)
    axes[1].set(title="Train tốt nhất chưa phải lựa chọn tốt nhất",xlabel="Số láng giềng k",ylabel="MSE (ms²)",xticks=ks)
    axes[1].legend();axes[1].grid(alpha=.18)
    fig.suptitle("Thí nghiệm chọn mô hình: điểm test không tham gia tìm k",fontsize=16,fontweight="bold")
    save(fig,"02-chon-mo-hinh")


def group_figure():
    groups,_=grouped_data()
    fig,axes=plt.subplots(1,2,figsize=(11.7,6.5),layout="constrained")
    for ax,mode,title in zip(axes,("row","group"),("Chia theo dòng: 12 user trùng train–test","Chia theo user: 0 user trùng train–test")):
        roles=np.full(len(groups),4)
        for role,part in enumerate(group_partitions(mode)):roles[part]=role
        ax.imshow(roles.reshape(12,6),cmap=CMAP,norm=NORM,aspect="auto")
        ax.set(title=title,xlabel="Quan sát thứ j trong cùng user",ylabel="User ID",xticks=range(6),yticks=range(12))
        ax.set_xticks(np.arange(-.5,6),minor=True);ax.set_yticks(np.arange(-.5,12),minor=True)
        ax.grid(which="minor",color="white",lw=1.4);ax.tick_params(which="minor",bottom=False,left=False)
    fig.suptitle("Cùng 72 dòng, cùng số mẫu 48/12/12, khác câu hỏi tổng quát hóa",fontsize=15,fontweight="bold")
    fig.legend(handles=[Patch(color=BLUE,label="Train"),Patch(color=ORANGE,label="Validation"),Patch(color=GREEN,label="Test")],
               loc="outside lower center",ncols=3)
    save(fig,"03-chia-nhom")


def time_figure():
    t,available,train,val,test,purged,waiting=temporal_partitions()
    corrected=np.full(14,4);corrected[train]=0;corrected[val]=1;corrected[test]=2;corrected[purged]=3
    original=corrected.copy();original[purged]=0
    fig,ax=plt.subplots(figsize=(13.2,4.8),layout="constrained")
    ax.imshow(np.vstack([original,corrected]),cmap=CMAP,norm=NORM,aspect="auto",extent=(.5,14.5,1.5,-.5))
    ax.set(yticks=[0,1],yticklabels=["Chưa kiểm nhãn sẵn","Đã loại nhãn tương lai"],
           xticks=t,xlabel="Thời điểm dự đoán t (nhãn chỉ có ở t+2)",xlim=(.5,14.5),ylim=(1.7,-1.45))
    ax.axvline(8.5,color="#334155",ls="--");ax.axvline(12.5,color="#334155",ls="--")
    for i in t:ax.axvline(i-.5,color="white",lw=1)
    ax.annotate("Nhãn của t=8 chỉ có ở t=10",xy=(10,-.55),xytext=(5,-1.02),
                arrowprops={"arrowstyle":"->","color":RED},color=RED,fontsize=11)
    ax.text(8.5,1.6,"Fit trước t=9",ha="center",fontsize=10)
    ax.text(12.5,1.6,"Chốt trước t=13",ha="center",fontsize=10)
    ax.set_title("Chia theo ngày vẫn cần kiểm tra thời điểm nhãn có sẵn",fontsize=16,fontweight="bold",pad=14)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in [(BLUE,"Train"),(ORANGE,"Validation"),(GREEN,"Test"),
                                                            (RED,"Loại khỏi train"),(GRAY,"Chờ nhãn validation")]],
               loc="outside lower center",ncols=5,fontsize=9)
    save(fig,"04-chia-thoi-gian")


def cv_figure():
    folds=cv_folds(6,SEED)
    roles=np.full((4,8),4)
    roles[:3,:6]=0
    for fold,indices in enumerate(folds):roles[fold,indices]=1
    roles[3,:6]=0;roles[3,6:]=2
    fig,ax=plt.subplots(figsize=(11.6,4.9),layout="constrained")
    ax.imshow(roles,cmap=CMAP,norm=NORM,aspect="auto")
    ax.set(xticks=range(8),xticklabels=[f"D{i}" for i in range(6)]+["T0","T1"],
           yticks=range(4),yticklabels=["CV fold 1","CV fold 2","CV fold 3","Refit và đánh giá cuối"],xlabel="Mẫu development D và test độc lập T")
    ax.set_xticks(np.arange(-.5,8),minor=True);ax.set_yticks(np.arange(-.5,4),minor=True)
    ax.grid(which="minor",color="white",lw=2);ax.tick_params(which="minor",bottom=False,left=False)
    ax.axvline(5.5,color="#0f172a",lw=2)
    ax.set_title("Mỗi fold fit lại toàn bộ tiền xử lý + mô hình; test nằm ngoài CV",fontsize=15,fontweight="bold",pad=16)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in [(BLUE,"Dùng để fit"),(ORANGE,"Validation fold"),
                                                            (GREEN,"Đánh giá cuối"),(GRAY,"Không dùng trong CV")]],
               loc="outside lower center",ncols=4,fontsize=10)
    save(fig,"05-cross-validation")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.facecolor":"white","savefig.facecolor":"white"})
    roles_figure();selection_figure();group_figure();time_figure();cv_figure()
    print(f"Generated 5 PNG and 5 SVG figures in {OUT}")


if __name__ == "__main__":
    main()
