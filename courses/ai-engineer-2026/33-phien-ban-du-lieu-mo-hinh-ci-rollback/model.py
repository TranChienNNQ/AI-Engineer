"""Fixed numeric logistic fixture; no pickle, training, network or side effect."""
import math
from data import FEATURES


def valid_vector(x):
    if not isinstance(x,list) or len(x)!=2:raise ValueError('shape')
    if any(type(v) not in {int,float} or not math.isfinite(v) or not 0<=v<=1 for v in x):raise ValueError('feature_range')


def score(model, preprocessing, x):
    valid_vector(x)
    if preprocessing!={'feature_order':FEATURES,'scale':[1.0,1.0]}:raise ValueError('preprocessing_contract')
    z=sum(w*v/s for w,v,s in zip(model['weights'],x,preprocessing['scale']))+model['bias']
    return 1/(1+math.exp(-z))


def validate_model(model, preprocessing, contract):
    if set(model)!={'format','weights','bias','threshold'} or model['format']!='logistic-v1':raise ValueError('model_format')
    if not isinstance(model['weights'],list) or len(model['weights'])!=2:raise ValueError('weights_shape')
    if any(type(v) not in {float,int} or not math.isfinite(v) or abs(v)>100 for v in model['weights']+[model['bias']]):raise ValueError('weights_value')
    if type(model['threshold']) not in {float,int} or not 0<model['threshold']<1:raise ValueError('threshold')
    if contract!={'input_schema':'ticket-v1','features':FEATURES,'output':'priority-0-or-1','label_mapping':{'0':'normal','1':'urgent'}}:
        raise ValueError('api_contract')
    score(model,preprocessing,[.5,0.0])


def metrics(model, preprocessing, rows):
    pairs=[(int(score(model,preprocessing,r['x'])>=model['threshold']),r['y']) for r in rows]
    tp=sum(a==b==1 for a,b in pairs);fp=sum(a==1 and b==0 for a,b in pairs)
    fn=sum(a==0 and b==1 for a,b in pairs);tn=sum(a==b==0 for a,b in pairs)
    return dict(n=len(pairs),tp=tp,fp=fp,fn=fn,tn=tn,accuracy=(tp+tn)/len(pairs) if pairs else None)


def evaluate(model, preprocessing, rows):
    return dict(overall=metrics(model,preprocessing,rows),
                slices={s:metrics(model,preprocessing,[r for r in rows if r['segment']==s]) for s in ['mobile','web']})
