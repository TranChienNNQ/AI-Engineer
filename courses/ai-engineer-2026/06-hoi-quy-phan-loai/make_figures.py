"""Generate five mathematical figures for lesson 6, as PNG and SVG."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab import (development_data, model_scores, select_model, sigmoid, softmax,
                 test_data, threshold_example)


OUT=Path(__file__).resolve().parent/"figures"
BLUE,ORANGE,GREEN,RED="#2563eb","#d97706","#15803d","#dc2626"


def save(fig,name):
    fig.savefig(OUT/f"{name}.png",dpi=170)
    fig.savefig(OUT/f"{name}.svg")
    plt.close(fig)


def regression_figure():
    train,val=development_data("regression")
    model,_=select_model("regression",train,val)
    tx,ty=test_data("regression")
    grid=np.linspace(-.2,5.2,200)[:,None]
    fig,axes=plt.subplots(1,2,figsize=(12.2,5),layout="constrained")
    for data,label,color,marker in [(train,"Train",BLUE,"o"),(val,"Validation",ORANGE,"s"),((tx,ty),"Test sau khi chốt",GREEN,"D")]:
        axes[0].scatter(data[0][:,0],data[1],label=label,color=color,marker=marker,s=45,zorder=4)
    axes[0].plot(grid[:,0],model_scores(model,grid),color=BLUE,lw=2,label="Mô hình được chọn")
    axes[0].axhline(train[1].mean(),color="#64748b",ls="--",label="Baseline mean train")
    axes[0].set(title="Hồi quy dự đoán đại lượng số",xlabel="Đặc trưng tải x",ylabel="Latency giả định y (ms)")
    axes[0].legend(fontsize=9)
    history=model["history"][:101]
    axes[1].semilogy(history[:,0],history[:,1],color=BLUE,lw=2)
    axes[1].set(title="100 bước đầu: train MSE",xlabel="Số lần cập nhật",ylabel="MSE (ms²), trục logarit")
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle("Hồi quy tuyến tính: fit train, chọn L2 trên validation, rồi đo test",fontsize=15,fontweight="bold")
    save(fig,"01-hoi-quy")


def sigmoid_figure():
    fig,axes=plt.subplots(1,2,figsize=(12.2,5.1),layout="constrained")
    z=np.linspace(-6,6,400)
    axes[0].plot(z,sigmoid(z),color=BLUE,lw=2.5)
    for threshold,color in [(.5,GREEN),(.8,RED)]:
        boundary=np.log(threshold/(1-threshold))
        axes[0].axhline(threshold,color=color,ls="--",alpha=.8)
        axes[0].axvline(boundary,color=color,ls=":",alpha=.8)
        axes[0].scatter([boundary],[threshold],color=color,s=45)
    axes[0].set(title="Logit → sigmoid → xác suất",xlabel="Logit z",ylabel="p = sigmoid(z)",ylim=(-.03,1.03))
    axes[0].text(-5.7,.84,"Ngưỡng p=0.8 → z≈1.386",color=RED,fontsize=10)
    axes[0].text(-5.7,.54,"Ngưỡng p=0.5 → z=0",color=GREEN,fontsize=10)
    axes[0].grid(alpha=.18)
    a=np.linspace(-3,3,250);X1,X2=np.meshgrid(a,a);Z=X1-X2
    image=axes[1].contourf(X1,X2,sigmoid(Z),levels=np.linspace(0,1,21),cmap="viridis")
    lines=axes[1].contour(X1,X2,Z,levels=[0,np.log(4)],colors=["white",RED],linewidths=2)
    axes[1].clabel(lines,fmt={0:"τ=0.5",np.log(4):"τ=0.8"},fontsize=10)
    axes[1].set(title="Logit = x₁ − x₂: biên quyết định là đường thẳng",xlabel="x₁ (không đơn vị)",ylabel="x₂ (không đơn vị)",aspect="equal")
    fig.colorbar(image,ax=axes[1],label="Xác suất p của mô hình minh họa",ticks=[0,.25,.5,.75,1])
    fig.suptitle("Đầu ra phi tuyến không đồng nghĩa biên quyết định phi tuyến",fontsize=15,fontweight="bold")
    save(fig,"02-sigmoid-bien-quyet-dinh")


def loss_figure():
    train,val=development_data("binary")
    model,_=select_model("binary",train,val)
    fig,axes=plt.subplots(1,2,figsize=(11.8,5),layout="constrained")
    p=np.linspace(.001,.999,500)
    axes[0].plot(p,-np.log(p),color=BLUE,label="Nhãn y=1: −ln p",lw=2)
    axes[0].plot(p,-np.log1p(-p),color=ORANGE,label="Nhãn y=0: −ln(1−p)",lw=2)
    axes[0].set(title="Tự tin sai bị phạt mạnh",xlabel="Xác suất dự đoán p cho lớp 1",ylabel="Log loss một mẫu (nat)",ylim=(0,7))
    axes[0].legend(fontsize=10)
    history=model["history"][:301]
    axes[1].plot(history[:,0],history[:,1],color=BLUE,lw=2)
    train_acc=np.mean((sigmoid(model_scores(model,train[0]))>=.5)==train[1])
    axes[1].text(.97,.92,f"Accuracy train cuối: {100*train_acc:g}%\nLoss và accuracy đo hai việc khác nhau",
                 ha="right",va="top",transform=axes[1].transAxes,fontsize=10)
    axes[1].set(title="Logistic regression: 300 bước đầu",xlabel="Số lần cập nhật",ylabel="Mean train log loss (nat)")
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle("Log loss hướng dẫn tối ưu xác suất, không chỉ nhãn sau ngưỡng",fontsize=15,fontweight="bold")
    save(fig,"03-log-loss")


def threshold_figure():
    y,p,rows=threshold_example()
    fig,axes=plt.subplots(1,2,figsize=(11.7,5),layout="constrained")
    for i,(target,prob) in enumerate(zip(y,p)):
        axes[0].scatter(prob,i,color=BLUE if target==0 else ORANGE,s=75)
        axes[0].text(prob+.03,i,f"y={target}",va="center",fontsize=10)
    for threshold,color in [(.3,GREEN),(.5,"#64748b"),(.7,RED)]:
        axes[0].axvline(threshold,color=color,ls="--",label=f"τ={threshold:g}")
    axes[0].set(title="Bốn xác suất validation được dựng sẵn",xlabel="p (p≥τ → gán nhãn 1)",ylabel="Mẫu",yticks=range(4),xlim=(0,1),ylim=(-.6,3.6))
    axes[0].legend(loc="upper left",fontsize=9)
    costs=[row[3] for row in rows]
    bars=axes[1].bar(["0.3","0.5","0.7"],costs,color=[GREEN,ORANGE,RED],width=.55)
    axes[1].bar_label(bars,padding=4)
    axes[1].set(title="Chi phí validation: FP + 3×FN",xlabel="Ngưỡng τ",ylabel="Tổng chi phí (đơn vị quy ước)",ylim=(0,8.5))
    fig.suptitle("Chọn ngưỡng là một quyết định riêng, dùng development thay vì test",fontsize=15,fontweight="bold")
    save(fig,"04-nguong-quyet-dinh")


def softmax_figure():
    z=np.log(np.array([1.,2.,3.]));names=["billing","technical","account"]
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.9),layout="constrained",sharey=True)
    for ax,values,title,color in [(axes[0],softmax(z),"Một nhãn trong ba lớp: softmax",BLUE),
                                   (axes[1],sigmoid(z),"Ba nhãn có thể đồng thời: các sigmoid",ORANGE)]:
        bars=ax.bar(names,values,color=color,width=.6)
        ax.bar_label(bars,labels=[f"{v:.3f}" for v in values],padding=4)
        ax.set(title=title,ylabel="Đầu ra theo lớp/nhãn",ylim=(0,1))
        ax.text(.5,.91,f"Tổng = {values.sum():.6f}",ha="center",transform=ax.transAxes)
    fig.suptitle("Cùng logits [0, ln 2, ln 3], khác giả định về nhãn",fontsize=16,fontweight="bold")
    save(fig,"05-softmax-nhieu-nhan")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.facecolor":"white","savefig.facecolor":"white"})
    regression_figure();sigmoid_figure();loss_figure();threshold_figure();softmax_figure()
    print(f"Generated 5 PNG and 5 SVG figures in {OUT}")


if __name__=="__main__":
    main()
