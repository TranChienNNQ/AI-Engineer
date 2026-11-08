"""Synthetic prediction and delayed-feedback records. No labels in prediction records."""
from dataclasses import dataclass,asdict
import math
import json
import re
from hashlib import sha256


def ident(value):
    return isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',value) is not None


@dataclass(frozen=True)
class Prediction:
    prediction_id:str
    model_version:str
    feature_version:str
    predicted_at:int
    segment:str
    feature_value:float
    score:float

    def __post_init__(self):
        if not all(ident(x) for x in [self.prediction_id,self.model_version,self.feature_version]):raise ValueError('invalid identity/version')
        if type(self.predicted_at) is not int or self.predicted_at<0:raise ValueError('invalid prediction time')
        if type(self.segment) is not str or self.segment not in {'mobile','web'}:raise ValueError('unknown segment')
        for x in [self.feature_value,self.score]:
            if type(x) not in {int,float} or not 0<=x<=1 or not math.isfinite(x):raise ValueError('invalid numeric value')


@dataclass(frozen=True)
class Feedback:
    feedback_id:str
    prediction_id:str
    model_version:str
    label:int
    label_event_at:int
    available_at:int

    def __post_init__(self):
        if not all(ident(x) for x in [self.feedback_id,self.prediction_id,self.model_version]):raise ValueError('invalid feedback identity')
        if type(self.label) is not int or self.label not in {0,1}:raise ValueError('binary integer label required')
        if type(self.label_event_at) is not int or type(self.available_at) is not int or not 0<=self.label_event_at<=self.available_at:
            raise ValueError('invalid feedback time')


def dataset():
    predictions=[];feedback=[]
    scores=[.9,.8,.7,.6,.55,.45,.4,.3,.2,.1]
    reference=[1,1,1,1,0,1,0,0,0,0]
    current=[0,0,0,0,1,0,0,0,0,0]
    for prefix,start,probs,labels in [('ref',0,scores,reference),('now',100,scores,current),
                                     ('shift',200,[.9]*8+[.1]*2,[1]*8+[0]*2)]:
        for i,(p,y) in enumerate(zip(probs,labels)):
            key=f'{prefix}-{i:02}';when=start+i
            predictions.append(Prediction(key,'toy-v1','features-v1',when,'mobile' if i<5 else 'web',p,p))
            # Wrong predictions in current cohort receive feedback much later: selection bias by design.
            delayed=prefix=='now' and int(p>=.5)!=y
            feedback.append(Feedback('fb-'+key,key,'toy-v1',y,when+5,when+(100 if delayed else 10)))
    return predictions,feedback


def dataset_hash():
    p,f=dataset();payload=dict(predictions=[asdict(x) for x in p],feedback=[asdict(x) for x in f])
    return sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
