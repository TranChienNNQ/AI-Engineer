"""Small explicit input contracts, not a public upload gateway."""
from pathlib import Path
import wave
import numpy as np
from PIL import Image


def patchify(image,patch=4):
    a=np.asarray(image)
    if a.ndim!=3 or a.shape[2]!=3 or min(a.shape)<1 or not np.isfinite(a).all():raise ValueError('finite HWC RGB required')
    if type(patch) is not int or patch<1 or a.shape[0]%patch or a.shape[1]%patch:raise ValueError('patch must divide H/W')
    h,w,c=a.shape
    return a.reshape(h//patch,patch,w//patch,patch,c).transpose(0,2,1,3,4).reshape(-1,patch*patch*c)


def stft_power(signal,sample_rate=8000,frame=128,hop=64):
    x=np.asarray(signal)
    if x.ndim!=1 or not np.issubdtype(x.dtype,np.floating) or not np.isfinite(x).all():raise ValueError('finite floating mono samples required')
    if any(type(v) is not int or v<1 for v in [sample_rate,frame,hop]) or frame<2 or hop>frame or len(x)<frame:raise ValueError('STFT parameters/length')
    starts=range(0,len(x)-frame+1,hop)
    windows=np.stack([x[s:s+frame] for s in starts])*np.hanning(frame)
    spectrum=np.fft.rfft(windows,axis=1)
    return np.abs(spectrum)**2,np.fft.rfftfreq(frame,1/sample_rate)


def save_png(path,image):
    a=np.asarray(image)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3:raise ValueError('uint8 RGB required')
    Image.fromarray(a).save(path)


def load_png(path,expected=(16,16)):
    with Image.open(path) as image:
        if image.mode!='RGB' or image.size!=expected:raise ValueError('expected RGB image with exact width/height')
        return np.array(image,dtype=np.uint8)


def write_wav(path,x,sample_rate=8000):
    x=np.asarray(x)
    if x.ndim!=1 or len(x)<1 or not np.isfinite(x).all() or np.max(np.abs(x))>1 or type(sample_rate) is not int or sample_rate<1:raise ValueError('normalized mono and positive sample rate required')
    pcm=np.round(x*32767).astype('<i2')
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(sample_rate);w.writeframes(pcm.tobytes())


def read_wav(path,expected_rate=8000,max_frames=8000):
    with wave.open(str(path),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getframerate()!=expected_rate or w.getcomptype()!='NONE':raise ValueError('expected mono PCM16 at declared sample rate')
        n=w.getnframes()
        if not 0<n<=max_frames:raise ValueError('frame budget')
        payload=w.readframes(n)
        if len(payload)!=2*n:raise ValueError('truncated PCM')
    return np.frombuffer(payload,dtype='<i2').astype(np.float64)/32767
