import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data import dataset,origins,fingerprint,SPEC,validate_grid
from models import BASELINES,fit_ridge,fit_mlp,predict,batch_forecast,net
from metrics import point,seasonal_scale,residual_quantiles,interval


def summary(model,data,ticks,scale):
    y,p=batch_forecast(model,data,ticks)
    return dict(overall=point(y,p,scale),by_horizon=[point(y[:,h],p[:,h],scale) for h in range(7)],
        origins=list(ticks),actual=y.tolist(),prediction=p.tolist())


def _plain(obj):
    if isinstance(obj,torch.Tensor):return dict(tensor=obj.detach().cpu().tolist(),dtype=str(obj.dtype),shape=list(obj.shape))
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,dict):return {k:_plain(v) for k,v in obj.items()}
    if isinstance(obj,list):return [_plain(v) for v in obj]
    return obj


def digest(payload):return hashlib.sha256(json.dumps(_plain(payload),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def pack(model):
    out={k:v for k,v in model.items() if k!='net'}
    out=_plain(out)
    if model['kind']=='mlp':out['state']={k:v.detach().cpu().clone() for k,v in model['net'].state_dict().items()}
    return out


def unpack(payload):
    m=copy.deepcopy(payload)
    if m['kind'] in BASELINES:return m
    if m['kind'] not in ['ridge','mlp'] or m['horizon'] not in [1,7] or not 35<=m['train_end']<=280:raise ValueError('model spec')
    h=m['horizon'];features=28+(3*(h+1) if m['known'] else 0)
    for key,n in [('x_mean',features),('x_std',features),('y_mean',h),('y_std',h)]:
        value=np.asarray(m['prep'][key],dtype=float)
        if value.shape!=(n,) or not np.isfinite(value).all() or key.endswith('std') and np.any(value<=0):raise ValueError('normalizer shape/finite/scale')
        m['prep'][key]=value
    if m['kind']=='ridge':
        m['coef']=np.asarray(m['coef'],dtype=float)
        if m['coef'].shape!=(features+1,h) or not np.isfinite(m['coef']).all():raise ValueError('coefficient shape/finite')
    else:
        network=net(features,h);expected=network.state_dict();state=m.pop('state')
        if set(state)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('neural state shape/dtype/finite')
        network.load_state_dict(state);network.eval();m['net']=network
    return m


def save_bundle(path,model,q,data_hash):
    payload=dict(spec=dict(SPEC),model=pack(model),q=np.asarray(q).tolist(),data_hash=data_hash,
        calibration_origins=origins(280,364),calibration_target_end=364)
    torch.save(dict(payload=payload,digest=digest(payload)),path)


def load_bundle(path,expected_data_hash=None):
    b=torch.load(path,map_location='cpu',weights_only=True)
    if set(b)!=set(['payload','digest']) or digest(b['payload'])!=b['digest']:raise ValueError('bundle digest')
    p=b['payload']
    if p['spec']!=SPEC or p['calibration_target_end']!=364 or p['calibration_origins']!=origins(280,364):raise ValueError('forecast/calibration spec mismatch')
    if expected_data_hash is not None and p['data_hash']!=expected_data_hash:raise ValueError('data snapshot mismatch')
    q=np.asarray(p['q'],dtype=float)
    if q.shape!=(7,) or not np.isfinite(q).all() or np.any(q<0):raise ValueError('interval width contract')
    return unpack(p['model']),q,p


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    data=dataset(True);stable=dataset(False);validate_grid(data)
    val_orig=origins(224,280);cal_orig=origins(280,364);test_orig=origins(364,420)
    val_scale=seasonal_scale(data['y'][:224]);test_scale=seasonal_scale(data['y'][:280]);development={};search={}
    for kind in BASELINES:development[kind]=dict(kind=kind,train_end=224)
    for name,known in [('ridge_lags',False),('ridge_known',True)]:
        trials=[];models=[]
        for lam in [.1,1.,10.]:
            m=fit_ridge(data,224,known,7,lam);mae=summary(m,data,val_orig,val_scale)['overall']['mae'];trials.append(dict(lam=lam,validation_mae=mae));models.append(m)
        best=min(range(3),key=lambda i:trials[i]['validation_mae']);development[name]=models[best];search[name]=trials
    development['ridge_recursive']=fit_ridge(data,224,True,1,1.,True)
    development['mlp_known'],history=fit_mlp(data,224,180,val_orig)
    validation={name:summary(m,data,val_orig,val_scale) for name,m in development.items()}
    selected=min(validation,key=lambda n:validation[n]['overall']['mae'])
    final={kind:dict(kind=kind,train_end=280) for kind in BASELINES}
    for name in ['ridge_lags','ridge_known','ridge_recursive']:
        old=development[name];final[name]=fit_ridge(data,280,old['known'],old['horizon'],old['lam'],old['recursive'])
    final['mlp_known'],_=fit_mlp(data,280,development['mlp_known']['epochs'])
    cal_y,cal_p=batch_forecast(final[selected],data,cal_orig);q,rank=residual_quantiles(cal_y-cal_p,.2)
    test={name:summary(m,data,test_orig,test_scale) for name,m in final.items()}
    stable_test={name:summary(m,stable,test_orig,test_scale) for name,m in final.items()}
    def interval_results(d):
        y,p=batch_forecast(final[selected],d,test_orig);lo=p-q;hi=p+q
        return dict(overall=interval(y,lo,hi,.2),by_horizon=[interval(y[:,h],lo[:,h],hi[:,h],.2) for h in range(7)],lower=lo.tolist(),upper=hi.tolist())
    intervals=dict(shift=interval_results(data),stable=interval_results(stable))
    save_bundle(out/'forecast.pt',final[selected],q,fingerprint(data));restored,qr,_=load_bundle(out/'forecast.pt',fingerprint(data))
    roundtrip=np.array_equal(q,qr) and all(np.array_equal(predict(final[selected],data,t),predict(restored,data,t)) for t in test_orig)
    if not roundtrip:raise AssertionError('inference roundtrip')
    return dict(seed=43,data_hash=fingerprint(data),stable_data_hash=fingerprint(stable),selected_by_validation=selected,
        validation=validation,ridge_search=search,mlp_history=history,mlp_selected_epoch=development['mlp_known']['epochs'],mlp_parameter_count=sum(v.numel() for v in development['mlp_known']['net'].parameters()),
        final_fit_end=280,initial_training_rows=190,final_training_rows=246,validation_scale=val_scale,test_scale=test_scale,
        test=test,stable_test=stable_test,calibration=dict(origins=cal_orig,actual=cal_y.tolist(),prediction=cal_p.tolist(),absolute_errors=np.abs(cal_y-cal_p).tolist(),q=q.tolist(),order_rank=rank,alpha=.2,rows_per_horizon=12),
        intervals=intervals,roundtrip_exact=roundtrip,bundle_bytes=(out/'forecast.pt').stat().st_size,
        scope='one synthetic daily series; rolling inputs with fixed weights after refit; no ARIMA/ETS library, Transformer/foundation model, adaptive conformal or production forecast service; 80% nominal widths have no asserted time-series coverage guarantee')
