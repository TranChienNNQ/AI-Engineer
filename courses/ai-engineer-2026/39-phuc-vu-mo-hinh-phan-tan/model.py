"""Small fixed-weight MLP for numerical equivalence, not a trained quality benchmark."""
import hashlib
import numpy as np
REVISION='mlp-v1'


def weights():
    # Factory: workers receive only their payload; no global NumPy copy of full weights.
    return dict(w1=np.array([[1,0,-1,2],[0,1,1,0],[1,-1,0,1],[0,2,1,-1]],dtype=np.float32),
                b1=np.array([0,1,0,0],dtype=np.float32),
                w2=np.array([[1,2],[-1,1],[2,0],[1,-2]],dtype=np.float32),b2=np.array([.5,-.5],dtype=np.float32))


def digest(payload):
    h=hashlib.sha256()
    for name,a in sorted(payload.items()):
        h.update(name.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def validate(x):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=4 or not 1<=x.shape[0]<=64 or not np.isfinite(x).all():raise ValueError('finite batch [1..64,4]')
    return x


def forward(x,w):return np.maximum(validate(x)@w['w1']+w['b1'],0)@w['w2']+w['b2']


def tp_parts(w):
    return [dict(w1=w['w1'][:,i:i+2].copy(),b1=w['b1'][i:i+2].copy(),w2=w['w2'][i:i+2].copy()) for i in [0,2]]


def pp_parts(w):return [dict(w1=w['w1'].copy(),b1=w['b1'].copy()),dict(w2=w['w2'].copy(),b2=w['b2'].copy())]


def payload_bytes(w):return sum(a.nbytes for a in w.values())
