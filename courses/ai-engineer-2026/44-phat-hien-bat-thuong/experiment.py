import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from data import tabular,temporal,fingerprint,TAB_SPEC,TIME_SPEC
from detectors import KINDS,fit,score,threshold,fit_temporal,temporal_scores
from metrics import confusion,average_precision,events
from alerts import apply


def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,list):return [plain(x) for x in v]
    return v


def digest(payload):return hashlib.sha256(json.dumps(plain(payload),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save_bundle(path,model,q,data_hash):
    payload=dict(spec=TAB_SPEC,model=plain(model),threshold=float(q),alpha=.05,comparison='>',data_hash=data_hash)
    Path(path).write_text(json.dumps(dict(payload=payload,digest=digest(payload)),allow_nan=False,indent=2)+'\n')


def load_bundle(path,expected_hash=None):
    b=json.loads(Path(path).read_text());p=b['payload']
    if digest(p)!=b['digest'] or p['spec']!=TAB_SPEC or p['comparison']!='>' or p['alpha']!=.05:raise ValueError('bundle digest/spec/policy')
    if expected_hash is not None and p['data_hash']!=expected_hash:raise ValueError('snapshot mismatch')
    q=p['threshold'];m=copy.deepcopy(p['model'])
    if not np.isfinite(q) or q<0 or m['kind'] not in KINDS or type(m['k']) is not int or m['k']!=5:raise ValueError('threshold/detector contract')
    shapes=dict(mean=(2,),std=(2,),median=(2,),madscale=(2,),reference=(300,2),projection=(2,1),inverse=(2,2))
    for key,shape in shapes.items():
        m[key]=np.asarray(m[key],dtype=float)
        if m[key].shape!=shape or not np.isfinite(m[key]).all():raise ValueError('model shape/finite')
    if np.any(m['std']<=0) or np.any(m['madscale']<=0) or not np.allclose(m['projection'].T@m['projection'],np.eye(1)) or not np.allclose(m['inverse'],m['inverse'].T) or np.min(np.linalg.eigvalsh(m['inverse']))<=0:raise ValueError('scale/projection/covariance contract')
    return m,float(q)


def evaluation(y,scores,q):return dict(point=confusion(y,scores>q),ap=average_precision(y,scores),scores=scores.tolist(),prediction=(scores>q).astype(int).tolist())


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);d=tabular();hash_=fingerprint(d,['x','label']);models={};tab={}
    for kind in KINDS:
        m=fit(d['x'][:300],kind);models[kind]=m;cal=score(m,d['x'][300:500]);q,rank=threshold(cal)
        val=evaluation(d['label'][500:620],score(m,d['x'][500:620]),q);test=evaluation(d['label'][620:],score(m,d['x'][620:]),q)
        test['recall_by_kind']={g:float(np.mean(np.asarray(test['prediction'])[d['kind'][620:]==g])) for g in ['off','along']}
        tab[kind]=dict(threshold=q,rank=rank,calibration_scores=cal.tolist(),validation=val,test=test)
    selected=max(KINDS,key=lambda k:tab[k]['validation']['point']['f1']);q=tab[selected]['threshold']
    save_bundle(out/'detector.json',models[selected],q,hash_);restored,qr=load_bundle(out/'detector.json',hash_)
    exact=np.array_equal(score(restored,d['x'][620:]),score(models[selected],d['x'][620:])) and q==qr
    assert exact,'inference bundle roundtrip'
    time=temporal();tm=fit_temporal(time);cal_scores,cal_state=temporal_scores(time,tm,168,336);test_scores,test_state=temporal_scores(time,tm,336,504,cal_state);stream={}
    for name,sc in test_scores.items():
        q,rank=threshold(cal_scores[name]);hits=sc>q;policy=apply(hits,start=336)
        stream[name]=dict(threshold=q,rank=rank,calibration_scores=cal_scores[name].tolist(),test=evaluation(time['label'][336:],sc,q),
            raw_events=events(time['label'][336:],np.flatnonzero(hits).astype(int)+336,start=336),
            alert_events=events(time['label'][336:],np.asarray(policy['firings'],dtype=int),start=336),policy=policy)
    # Chunked replay: detector and alert policy state are both required. Last-24 actuals are supplied by this offline data fixture.
    a,state=temporal_scores(time,tm,336,430,cal_state);b,continued=temporal_scores(time,tm,430,504,state)
    temporal_exact=all(np.array_equal(np.r_[a[k],b[k]],test_scores[k]) for k in stream) and continued==test_state
    q=stream['calendar_residual']['threshold'];p1=apply(a['calendar_residual']>q,start=336)
    (out/'stream-state.json').write_text(json.dumps(dict(cusum_state=state,policy=p1['state'],next_tick=430)))
    saved=json.loads((out/'stream-state.json').read_text())
    replay,replay_state=temporal_scores(time,tm,saved['next_tick'],504,saved['cusum_state'])
    assert replay_state==continued and all(np.array_equal(replay[k],b[k]) for k in b)
    p2=apply(replay['calendar_residual']>q,start=saved['next_tick'],state=saved['policy'])
    assert temporal_exact and p1['firings']+p2['firings']==stream['calendar_residual']['policy']['firings']
    return dict(seed=44,tabular=dict(spec=TAB_SPEC,data_hash=hash_,selected_by_validation_f1=selected,detectors=tab,roundtrip_exact=exact),
        temporal=dict(spec=TIME_SPEC,data_hash=fingerprint(time,['t','y','plan','plan_available_at','label']),fit=plain(tm),calibration_cusum_state=cal_state,final_cusum_state=test_state,detectors=stream,chunk_replay_exact=temporal_exact,policy_replay_exact=True,policy=dict(persist=2,clear=3,cooldown=4)),
        scope='synthetic novelty references assumed normal; NumPy only; no IsolationForest/LOF/OCSVM/deep model training or production alert service; empirical thresholds do not assert FPR guarantees')
