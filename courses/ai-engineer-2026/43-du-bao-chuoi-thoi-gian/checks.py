import copy
import math
from pathlib import Path
import numpy as np
import torch
from data import dataset,origins,fingerprint,validate_grid
from features import row,supervised,normalize_fit
from models import baseline,fit_ridge,fit_mlp,predict,BASELINES
from metrics import point,seasonal_scale,residual_quantiles,interval
from experiment import pack,unpack,save_bundle,load_bundle,digest
from arithmetic import hand


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    names=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        names.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,TypeError,RuntimeError):check(name,True)
        else:raise AssertionError(name+' accepted')
    d=dataset();validate_grid(d);stable=dataset(False)
    check('stable/shift identical before test',np.array_equal(d['y'][:364],stable['y'][:364]))
    check('test regimes actually differ',not np.array_equal(d['y'][364:],stable['y'][364:]))
    check('origin counts',[len(origins(a,b)) for a,b in [(224,280),(280,364),(364,420)]]==[8,12,8])
    check('origins/targets partition',origins(224,280)[0]==223 and origins(224,280)[-1]+7==279 and origins(280,364)[-1]+7==363)
    reject('partial evaluation block rejected',lambda:origins(224,281))
    bad=copy.deepcopy(d);bad['t'][10]=9;reject('duplicate time rejected',lambda:validate_grid(bad))
    bad=copy.deepcopy(d);bad['y'][10]=np.nan;reject('missing observation rejected',lambda:validate_grid(bad))
    y=np.arange(28,dtype=float)
    check('naive repeat',baseline(y,3,'naive').tolist()==[27.,27.,27.])
    check('mean28',baseline(y,3,'mean28').tolist()==[13.5]*3)
    check('drift extrapolation',baseline(y,3,'drift').tolist()==[28.,29.,30.])
    check('seasonal naive wrap',np.array_equal(baseline(y,14,'seasonal_naive'),np.r_[y[-7:],y[-7:]]))
    reject('baseline short history',lambda:baseline([1.,2.]))
    x,target,ticks=supervised(d,224);check('direct supervised rows',x.shape==(190,52) and target.shape==(190,7))
    check('last training label cutoff',int(ticks[-1])==216 and int(ticks[-1])+7==223 and np.all(ticks+7<224))
    check('lag-only dimension',row(d,223,known=False).shape==(28,))
    check('relative lag anchor',row(d,223)[27]==0 and np.array_equal(row(d,223)[:28],d['y'][196:224]-d['y'][223]))
    changed=copy.deepcopy(d);changed['y'][224:]+=100000
    x2,y2,t2=supervised(changed,224)
    check('future targets cannot alter fit matrices',np.array_equal(x,x2) and np.array_equal(target,y2) and np.array_equal(ticks,t2))
    p=normalize_fit(x,target);p2=normalize_fit(x2,y2)
    check('scaler train-only identity',all(np.array_equal(p[k],p2[k]) for k in p))
    bad=copy.deepcopy(d);bad['plan_available_at'][225]=224
    reject('future covariate unavailable rejected',lambda:row(bad,223))
    check('lag model ignores unprovided plan',np.array_equal(row(bad,223,known=False),row(d,223,known=False)))
    forecast_plan=copy.deepcopy(d);forecast_plan['plan'][224]=1-d['plan'][224]
    check('known planned future is a legitimate feature',not np.array_equal(row(forecast_plan,223),row(d,223)))
    reject('NaN history rejected',lambda:row(dict(d,y=np.r_[d['y'][:223],np.nan,d['y'][224:]]),223))
    reject('future origin bounds',lambda:row(d,418,7))
    ridge=fit_ridge(d,224);coef=ridge['coef'];a=np.c_[np.ones(len(x)),(x-p['x_mean'])/p['x_std']];yn=(target-p['y_mean'])/p['y_std']
    normal=a.T@(a@coef-yn)+np.r_[np.zeros((1,7)),coef[1:]]
    check('ridge normal equation residual',np.max(np.abs(normal))<1e-8)
    check('ridge intercept unpenalized',np.max(np.abs((a@coef-yn).mean(axis=0)))<1e-10)
    mlp,history=fit_mlp(d,224,3,origins(224,280))
    check('MLP actual train trace',len(history)==3 and all(np.isfinite(v['normalized_train_mse']) for v in history))
    check('MLP gradients finite nonzero',all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0 for v in mlp['net'].parameters()))
    rec=fit_ridge(d,224,True,1,1,True)
    models={k:dict(kind=k,train_end=224) for k in BASELINES};models.update(ridge=ridge,recursive=rec,mlp=mlp)
    for name,m in models.items():
        a1=predict(m,d,223);a2=predict(m,changed,223)
        check(name+' future-y invariant',np.array_equal(a1,a2) and a1.shape==(7,) and np.isfinite(a1).all())
    # Independent recursive rollout using explicit append of each prediction.
    buffer=d['y'][:224].copy();reference=[]
    from models import _predict_row
    for step in range(7):
        feature=row(d,223+step,1,True,history=buffer,asof=223);value=float(_predict_row(rec,feature,buffer[-1])[0]);reference.append(value);buffer=np.r_[buffer,value]
    check('recursive generated-history rollout',np.array_equal(reference,predict(rec,d,223)))
    scale=seasonal_scale(d['y'][:224]);check('MASE scale explicit loop',abs(scale-sum(abs(d['y'][i]-d['y'][i-7]) for i in range(7,224))/217)<1e-12)
    reject('zero MASE denominator rejected',lambda:seasonal_scale(np.ones(20)))
    m=point([0,10,20],[2,8,25],2)
    check('MAE/RMSE hand',m['mae']==3 and abs(m['rmse']-math.sqrt(11))<1e-12)
    check('MASE/WAPE hand',m['mase']==1.5 and m['wape']==.3)
    check('bias sign',abs(m['bias']-5/3)<1e-12)
    check('WAPE zero denominator explicit',point([0,0],[1,2],1)['wape'] is None)
    reject('metric shape mismatch',lambda:point([1,2],[1],1))
    reject('metric NaN',lambda:point([1],[np.nan],1))
    errors=np.arange(1,10,dtype=float)[:,None];q,rank=residual_quantiles(errors,.2)
    check('finite order-statistic hand',rank==8 and q.tolist()==[8.])
    q12,rank12=residual_quantiles(np.tile(np.arange(1,13)[:,None],(1,7)),.2)
    check('12 calibration rows rank11',rank12==11 and q12.tolist()==[11.]*7)
    reject('too small calibration',lambda:residual_quantiles(np.ones((2,7)),.1))
    reject('bad alpha',lambda:residual_quantiles(np.ones((12,7)),1.))
    s=interval([5.],[0.],[4.],.2);check('interval score hand',s==dict(coverage=0.,mean_width=4.,interval_score=14.))
    check('inclusive interval boundary',interval([4],[0],[4],.2)['coverage']==1.)
    reject('crossed interval rejected',lambda:interval([1],[2],[0]))
    for name,m in [('ridge',ridge),('mlp',mlp),('recursive',rec),('naive',models['naive'])]:
        restored=unpack(pack(m));check(name+' model serialization exact',np.array_equal(predict(m,d,223),predict(restored,d,223)))
    save_bundle(out/'test.pt',ridge,np.ones(7),fingerprint(d));loaded,q,_=load_bundle(out/'test.pt',fingerprint(d))
    check('bundle point/width exact',np.array_equal(q,np.ones(7)) and np.array_equal(predict(loaded,d,223),predict(ridge,d,223)))
    reject('bundle snapshot mismatch',lambda:load_bundle(out/'test.pt','wrong'))
    b=torch.load(out/'test.pt',weights_only=True);b['payload']['q'][0]=5;torch.save(b,out/'bad.pt')
    reject('bundle digest mismatch',lambda:load_bundle(out/'bad.pt'))
    b=torch.load(out/'test.pt',weights_only=True);b['payload']['model']['prep']['x_std']=[0]*52;b['digest']=digest(b['payload']);torch.save(b,out/'bad.pt')
    reject('bundle normalizer invalid even with new digest',lambda:load_bundle(out/'bad.pt'))
    check('hand training window count',hand()['windows']['rows']==190)
    return dict(checks=len(names),passed=len(names),names=names)
