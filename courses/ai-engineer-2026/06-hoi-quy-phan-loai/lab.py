"""Lesson 6: linear regression and binary logistic regression using NumPy.

Synthetic data, full-batch gradient descent, train-only standardization.
Choose L2 on validation, then evaluate only the chosen model on test.
Run python3 lab.py --mode all. Softmax examples do not train a multiclass model.
"""

import argparse
import math

import numpy as np


def sigmoid(z):
    """Stable sigmoid; avoid exp(1000) for large negative logits."""
    z=np.asarray(z,dtype=np.float64)
    out=np.empty_like(z)
    positive=z>=0
    out[positive]=1/(1+np.exp(-z[positive]))
    ez=np.exp(z[~positive])
    out[~positive]=ez/(1+ez)
    return out


def data_loss(kind,y,z):
    y,z=np.asarray(y),np.asarray(z)
    if y.ndim!=1 or y.shape!=z.shape or not len(y):
        raise ValueError("Use matching nonempty 1-D targets and scores.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(z)):
        raise ValueError("Use finite targets and scores.")
    if kind=="regression":
        return float(np.mean((z-y)**2))
    if kind=="binary":
        if not np.all((y==0)|(y==1)):
            raise ValueError("Binary targets must be 0 or 1.")
        # y=1: softplus(-z); y=0: softplus(z). Avoid subtracting two large terms.
        return float(np.mean(np.logaddexp(0,np.where(y==1,-z,z))))
    raise ValueError("Use regression or binary.")


def loss_and_gradients(kind,X,y,w,b,l2=0.0):
    """Objective = mean data loss + l2/2 * ||w||^2; bias is not penalized."""
    if X.ndim!=2 or w.ndim!=1 or X.shape!=(len(y),len(w)):
        raise ValueError("Expected X=(n,d), y=(n,), w=(d,).")
    if l2<0 or not math.isfinite(l2):
        raise ValueError("L2 coefficient must be finite and nonnegative.")
    z=X@w+b
    loss=data_loss(kind,y,z)
    error=z-y if kind=="regression" else sigmoid(z)-y
    factor=2.0 if kind=="regression" else 1.0
    dw=factor/len(y)*(X.T@error)+l2*w
    db=float(factor*np.mean(error))
    objective=loss+0.5*l2*float(w@w)
    return loss,objective,dw,db


def fit_model(kind,X,y,l2=0.0,lr=0.1,steps=3000):
    """Fit feature statistics and parameters using only supplied training rows."""
    X,y=np.asarray(X,dtype=float),np.asarray(y,dtype=float)
    if X.ndim!=2 or y.ndim!=1 or X.shape[0]!=len(y) or len(y)==0:
        raise ValueError("Expected nonempty X=(n,d) and y=(n,).")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Training values must be finite.")
    if not math.isfinite(lr) or lr<=0 or steps<0:
        raise ValueError("Use positive finite lr and nonnegative steps.")
    mean=X.mean(axis=0)
    scale=X.std(axis=0,ddof=0)
    scale=np.where(scale>0,scale,1.0)
    A=(X-mean)/scale
    w,b=np.zeros(X.shape[1]),0.0
    history=[]
    for step in range(steps+1):
        loss,obj,dw,db=loss_and_gradients(kind,A,y,w,b,l2)
        history.append((step,loss,obj))
        if step<steps:
            w,b=w-lr*dw,b-lr*db
    return {"kind":kind,"w":w,"b":b,"mean":mean,"scale":scale,
            "l2":l2,"history":np.array(history)}


def model_scores(model,X):
    """Raw numeric prediction for regression; logits for binary classification."""
    return ((X-model["mean"])/model["scale"])@model["w"]+model["b"]


def development_data(kind):
    if kind=="regression":
        train=(np.arange(6,dtype=float)[:,None],np.array([10.,12.,11.,14.,13.,15.]))
        vx=np.array([.5,2.5,4.5])
        return train,(vx[:,None],10+vx)
    if kind=="binary":
        x=np.array([-3.,-2.,-1.,-.5,.5,1.,2.,3.])
        y=np.array([0.,0.,1.,0.,1.,0.,1.,1.])
        vx=np.array([-2.5,-1.5,.25,1.5,2.5])
        return (x[:,None],y),(vx[:,None],np.array([0.,0.,0.,1.,1.]))
    raise ValueError("Use regression or binary.")


def test_data(kind):
    if kind=="regression":
        x=np.array([1.5,3.5])
        return x[:,None],10+x
    if kind=="binary":
        return np.array([-2.2,-.8,.8,2.2])[:,None],np.array([0.,0.,1.,1.])
    raise ValueError("Use regression or binary.")


def select_model(kind,train,val,candidates=(0.0,0.1,1.0),lr=0.1,steps=3000):
    """No test-data argument. Compare predictive validation loss, not penalty."""
    X,y=train;vx,vy=val
    rows=[];best=None;best_loss=float("inf")
    for l2 in candidates:
        model=fit_model(kind,X,y,l2,lr,steps)
        val_loss=data_loss(kind,vy,model_scores(model,vx))
        rows.append((l2,model["history"][-1,1],val_loss))
        if val_loss<best_loss:
            best,best_loss=model,val_loss
    return best,rows


def softmax(z):
    """Normalize the final axis, with a max shift for numerical stability."""
    z=np.asarray(z,dtype=np.float64)
    if z.ndim<1 or z.shape[-1]==0 or not np.all(np.isfinite(z)):
        raise ValueError("Use finite logits with a nonempty class axis.")
    shifted=z-np.max(z,axis=-1,keepdims=True)
    ez=np.exp(shifted)
    return ez/np.sum(ez,axis=-1,keepdims=True)


def categorical_loss(z,true_index):
    """Single-example multiclass negative log likelihood from a 1-D logit vector."""
    z=np.asarray(z,dtype=float)
    if z.ndim!=1 or not 0<=true_index<len(z):
        raise ValueError("Use a 1-D logit vector and a valid true-class index.")
    shifted=z-np.max(z)
    return float(np.log(np.sum(np.exp(shifted)))-shifted[true_index])


def threshold_example():
    # Constructed validation scores; separate from the fitted logistic demo.
    y=np.array([0,1,1,0]);p=np.array([.2,.4,.6,.8])
    rows=[]
    for threshold in (.3,.5,.7):
        pred=p>=threshold
        fp=int(np.sum(pred & (y==0)))
        fn=int(np.sum(~pred & (y==1)))
        rows.append((threshold,fp,fn,fp+3*fn))
    return y,p,rows


def hand_demo():
    for kind,X,y,lr in [("regression",np.array([[1.],[2.]]),np.array([3.,5.]),.1),
                         ("binary",np.array([[-1.],[1.]]),np.array([0.,1.]),1.)]:
        w,b=np.zeros(1),0.0
        old,_,dw,db=loss_and_gradients(kind,X,y,w,b)
        w,b=w-lr*dw,b-lr*db
        z=X@w+b
        print(f"{kind}: old_loss={old:.6f}; dw={dw}; db={db:.6f}")
        print(f"new w={w}; b={b:.6f}; new loss={data_loss(kind,y,z):.6f}")
        print("new outputs:",z if kind=="regression" else sigmoid(z))


def training_demo(kind,lr,steps):
    train,val=development_data(kind)
    model,rows=select_model(kind,train,val,lr=lr,steps=steps)
    print("L2      train_data_loss    validation_data_loss")
    for l2,tr,va in rows:print(f"{l2:<7g} {tr:<18.6f} {va:.6f}")
    print(f"Selected and frozen L2={model['l2']:g}; no train+validation refit")
    raw_w=model["w"]/model["scale"]
    raw_b=model["b"]-float(raw_w@model["mean"])
    print("raw-feature coefficients:",raw_w,"; intercept:",raw_b)
    # Final test data are accessed only after model selection.
    tx,ty=test_data(kind);z=model_scores(model,tx)
    loss=data_loss(kind,ty,z)
    prior=float(np.mean(train[1]))
    if kind=="regression":
        print("test predictions:",z)
        print(f"test MSE={loss:.6f}; RMSE={math.sqrt(loss):.6f}; MAE={np.mean(np.abs(z-ty)):.6f}")
        print(f"train-mean baseline test MSE={np.mean((prior-ty)**2):.6f}")
    else:
        p=sigmoid(z)
        baseline_logit=math.log(prior/(1-prior))  # This constructed train set has both classes.
        print("test probabilities:",p)
        print(f"test log loss={loss:.6f}; accuracy at 0.5={np.mean((p>=.5)==ty):.6f}")
        print(f"train-prior baseline test log loss={data_loss('binary',ty,np.full(len(ty),baseline_logit)):.6f}")


def threshold_demo():
    _,_,rows=threshold_example()
    for threshold,fp,fn,cost in rows:
        print(f"threshold={threshold:.1f}; FP={fp}; FN={fn}; validation cost={cost}")
    print("Selected validation threshold:",min(rows,key=lambda row:row[3])[0])


def multiclass_demo():
    z=np.log(np.array([1.,2.,3.]))
    p=softmax(z)
    print("class order: billing, technical, account")
    print("softmax:",p,"; sum:",p.sum())
    print("true class=account; categorical loss:",categorical_loss(z,2))
    print("independent sigmoids:",sigmoid(z),"; sum:",sigmoid(z).sum())


def stability_demo():
    z=np.array([-1000.,0.,1000.]);y=np.array([1.,0.,0.])
    print("extreme logits:",z,"; sigmoid:",sigmoid(z))
    print("finite mean BCE from logits:",data_loss("binary",y,z))
    print("softmax with +1000 shift:",softmax(np.log([1.,2.,3.])+1000))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["all","hand","regression","classification","threshold","multiclass","stability"],default="all")
    parser.add_argument("--lr",type=float,default=.1)
    parser.add_argument("--steps",type=int,default=3000)
    args=parser.parse_args()
    if not math.isfinite(args.lr) or args.lr<=0 or args.steps<0:
        parser.error("Use finite positive --lr and nonnegative --steps.")
    np.set_printoptions(precision=6,suppress=True)
    demos={"hand":hand_demo,"regression":lambda:training_demo("regression",args.lr,args.steps),
           "classification":lambda:training_demo("binary",args.lr,args.steps),"threshold":threshold_demo,
           "multiclass":multiclass_demo,"stability":stability_demo}
    for name,demo in demos.items():
        if args.mode in ("all",name):
            print(f"\n--- {name} ---");demo()


if __name__=="__main__":
    main()
