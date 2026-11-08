"""Synthetic labels link color/orientation to tone/envelope by an artificial codebook."""
import hashlib
import numpy as np
from media import save_png,write_wav

COLORS=['đỏ','lục','lam'];DIRECTIONS=['dọc','ngang'];FREQUENCIES=[300,600,1200]
CAPTIONS=[f'{c} {d}' for c in COLORS for d in DIRECTIONS]
SPEC=dict(version='synthetic-trimodal-v1',height=16,width=16,channels=3,patch=4,sample_rate=8000,samples=800,frame=128,hop=64,log_power='log1p',embedding=12)


def make_record(label,index):
    rng=np.random.default_rng(41000+label*100+index);color,direction=divmod(label,2)
    yy,xx=np.indices((16,16));coord=xx if direction==0 else yy
    bars=((coord+int(rng.integers(0,8)))%8<4).astype(float)
    image=np.full((16,16,3),.04);image[:,:,color]+=(.55+rng.uniform(0,.25))*bars
    image=np.round(np.clip(image+rng.normal(0,.015,image.shape),0,1)*255).astype(np.uint8)
    t=np.arange(800)/8000;envelope=np.linspace(.2,.85,800)
    if direction==1:envelope=envelope[::-1]
    freq=FREQUENCIES[color]*(1+rng.uniform(-.01,.01));phase=rng.uniform(0,2*np.pi)
    audio=(.8+rng.uniform(-.1,.1))*envelope*np.sin(2*np.pi*freq*t+phase)+rng.normal(0,.006,800)
    return dict(id=f'c{label}-g{index:02d}',label=label,text=CAPTIONS[label],image=image,audio=audio.astype(np.float64))


def dataset():
    # Each record/group owns all its modalities. A group never spans split boundaries.
    return {name:[make_record(c,i) for c in range(6) for i in indices] for name,indices in [('train',range(16)),('validation',range(16,20)),('test',range(20,24))]}


def fingerprint(splits):
    h=hashlib.sha256()
    for split,records in splits.items():
        h.update(split.encode())
        for r in records:
            h.update((r['id']+'|'+str(r['label'])+'|'+r['text']).encode());h.update(r['image'].tobytes());h.update(r['audio'].tobytes())
    return h.hexdigest()


def export_examples(out):
    from pathlib import Path
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for label,name in [(0,'red-vertical'),(5,'blue-horizontal')]:
        r=make_record(label,20);save_png(out/(name+'.png'),r['image']);write_wav(out/(name+'.wav'),r['audio'])
    return dict(files=4,scope='synthetic RGB stripes and sine tones, not photographs or speech')


def rule_predict(records,modality):
    output=[]
    for r in records:
        if modality=='image':
            a=r['image'].astype(float);color=int(a.mean(axis=(0,1)).argmax())
            dx=np.abs(np.diff(a,axis=1)).mean();dy=np.abs(np.diff(a,axis=0)).mean();direction=int(dy>dx)
        elif modality=='audio':
            x=r['audio'];peak=np.fft.rfftfreq(len(x),1/8000)[np.abs(np.fft.rfft(x)).argmax()]
            color=int(np.argmin(np.abs(np.array(FREQUENCIES)-peak)))
            direction=int(np.mean(x[-200:]**2)<np.mean(x[:200]**2))
        else:raise ValueError('rule modality')
        output.append(2*color+direction)
    return np.array(output)
