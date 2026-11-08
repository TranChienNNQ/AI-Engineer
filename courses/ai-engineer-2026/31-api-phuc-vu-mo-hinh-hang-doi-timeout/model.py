"""A real numeric forward pass with fixed teaching weights; no training/quality claim."""
from dataclasses import dataclass
from hashlib import sha256
import asyncio
import json
import math
import copy
from collections import deque


@dataclass(frozen=True)
class Features:
    latency_ratio: float
    error_ratio: float

    def __post_init__(self):
        for value in [self.latency_ratio,self.error_ratio]:
            if type(value) not in {int,float} or not 0<=value<=1 or not math.isfinite(value):
                raise ValueError('features must be finite numbers in [0,1], not bool/string')


ARTIFACT=dict(model_version='toy-logit-v1',feature_order=['latency_ratio','error_ratio'],
              weights=[2.0,3.0],bias=-2.0,threshold=.5)
ARTIFACT_HASH=sha256(json.dumps(ARTIFACT,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class Model:
    def __init__(self):
        self.artifact=copy.deepcopy(ARTIFACT)
        self.artifact_hash=sha256(json.dumps(self.artifact,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def predict(self,features):
        values=[getattr(features,name) for name in self.artifact['feature_order']]
        z=sum(w*x for w,x in zip(self.artifact['weights'],values))+self.artifact['bias']
        probability=1/(1+math.exp(-z))
        return dict(model_version=self.artifact['model_version'],artifact_hash=self.artifact_hash,
                    priority_score=probability,label='high' if probability>=self.artifact['threshold'] else 'normal')


class Backend:
    """Cooperative async delay/fault adapter plus the numeric model, not CPU/GPU load."""
    def __init__(self,delay=.005,gate=None,fail_ids=()):
        self.delay=delay;self.gate=gate;self.fail_ids=set(fail_ids)
        self.started=asyncio.Queue(maxsize=128)
        self.calls=deque(maxlen=128);self.finished=deque(maxlen=128);self.cancelled=deque(maxlen=128)
        self.call_count=0
        self.model=None;self.load_count=0

    async def load(self):
        self.model=Model();self.load_count+=1
        self.model.predict(Features(0.0,0.0))  # Minimal warm-up, no training.

    async def infer(self,features,job_id):
        self.calls.append(job_id);self.call_count+=1
        if self.started.full():
            self.started.get_nowait();self.started.task_done()
        self.started.put_nowait(job_id)
        try:
            if self.gate is not None:await self.gate.wait()
            await asyncio.sleep(self.delay)
            if job_id in self.fail_ids:raise RuntimeError('synthetic inference failure')
            result=self.model.predict(features);self.finished.append(job_id);return result
        except asyncio.CancelledError:
            self.cancelled.append(job_id);raise

    async def close(self):
        self.model=None
        while not self.started.empty():
            self.started.get_nowait();self.started.task_done()
