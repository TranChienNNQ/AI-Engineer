import copy
import numpy as np
import torch
from torch import nn
from data import SPEC
from features import row,supervised,normalize_fit

BASELINES=['naive','seasonal_naive','drift','mean28']


def baseline(history,horizon=7,kind='naive'):
    y=np.asarray(history,dtype=float)
    if y.ndim!=1 or len(y)<28 or not np.isfinite(y).all() or type(horizon) is not int or horizon<1:raise ValueError('baseline history/horizon')
    if kind=='naive':return np.repeat(y[-1],horizon)
    if kind=='seasonal_naive':return np.array([y[-7+(h%7)] for h in range(horizon)])
    if kind=='mean28':return np.repeat(y[-28:].mean(),horizon)
    if kind=='drift':return y[-1]+np.arange(1,horizon+1)*(y[-1]-y[0])/(len(y)-1)
    raise ValueError('baseline kind')


def fit_ridge(data,end,known=True,horizon=7,lam=1.,recursive=False):
    if not np.isfinite(lam) or lam<=0 or recursive and horizon!=1:raise ValueError('ridge config')
    x,y,ticks=supervised(data,end,horizon,known);prep=normalize_fit(x,y)
    z=(x-prep['x_mean'])/prep['x_std'];target=(y-prep['y_mean'])/prep['y_std'];a=np.c_[np.ones(len(z)),z]
    penalty=np.eye(a.shape[1])*lam;penalty[0,0]=0
    coef=np.linalg.solve(a.T@a+penalty,a.T@target)
    return dict(kind='ridge',known=known,horizon=horizon,recursive=recursive,train_end=end,lam=float(lam),prep=prep,coef=coef,fit_origins=ticks)


def net(features,horizon=7):return nn.Sequential(nn.Linear(features,32),nn.Tanh(),nn.Linear(32,horizon)).double()


def fit_mlp(data,end,epochs=180,validation_origins=None):
    torch.manual_seed(43);torch.set_num_threads(1)
    x,y,ticks=supervised(data,end,7,True);prep=normalize_fit(x,y);z=torch.tensor((x-prep['x_mean'])/prep['x_std']);target=torch.tensor((y-prep['y_mean'])/prep['y_std'])
    network=net(x.shape[1]);opt=torch.optim.Adam(network.parameters(),lr=.01)
    model=dict(kind='mlp',known=True,horizon=7,recursive=False,train_end=end,prep=prep,net=network,fit_origins=ticks)
    best=float('inf');best_state=None;best_epoch=epochs;history=[]
    for epoch in range(1,epochs+1):
        opt.zero_grad(set_to_none=True);loss=(network(z)-target).square().mean();loss.backward();opt.step()
        if validation_origins is not None:
            pred=np.stack([predict(model,data,t) for t in validation_origins]);truth=np.stack([data['y'][t+1:t+8] for t in validation_origins]);mae=float(np.abs(pred-truth).mean())
            history.append(dict(epoch=epoch,normalized_train_mse=float(loss.detach()),validation_mae=mae))
            if mae<best:best=mae;best_epoch=epoch;best_state=copy.deepcopy(network.state_dict())
    if best_state is not None:network.load_state_dict(best_state)
    model['epochs']=best_epoch;model['net'].eval();return model,history


def _predict_row(model,x,anchor):
    prep=model['prep'];z=(x-prep['x_mean'])/prep['x_std']
    if model['kind']=='ridge':normalized=np.r_[1.,z]@model['coef']
    else:
        with torch.no_grad():normalized=model['net'](torch.tensor(z)).numpy()
    return anchor+normalized*prep['y_std']+prep['y_mean']


def predict(model,data,origin):
    if model['kind'] in BASELINES:return baseline(data['y'][:origin+1],7,model['kind'])
    if model.get('recursive'):
        history=np.array(data['y'][:origin+1],copy=True);result=[]
        for step in range(7):
            tick=origin+step;x=row(data,tick,1,model['known'],history=history,asof=origin)
            value=float(_predict_row(model,x,history[-1])[0]);result.append(value);history=np.r_[history,value]
        return np.array(result)
    x=row(data,origin,model['horizon'],model['known'])
    return _predict_row(model,x,float(data['y'][origin]))


def batch_forecast(model,data,ticks):
    prediction=np.stack([predict(model,data,int(t)) for t in ticks]);actual=np.stack([data['y'][t+1:t+8] for t in ticks])
    return actual,prediction
