import copy
import cmath
import math
from pathlib import Path
import wave
import numpy as np
from PIL import Image
import torch
from media import patchify,stft_power,save_png,load_png,write_wav,read_wav
from data import dataset,SPEC,fingerprint,CAPTIONS,rule_predict
from preprocess import fit,transform,bow,words
from model import Align,positive_set_loss,total_loss
from training import save_bundle,load_bundle,state_hash,tensor_features
from metrics import late_fuse,top1
from arithmetic import hand


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.manual_seed(410);torch.set_num_threads(1)
    names=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        names.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,TypeError,RuntimeError):check(name,True)
        else:raise AssertionError(name+' accepted')
    for h,w,p in [(4,4,2),(8,12,4),(16,16,4)]:
        a=np.arange(h*w*3).reshape(h,w,3);got=patchify(a,p)
        ref=np.array([a[i:i+p,j:j+p].ravel() for i in range(0,h,p) for j in range(0,w,p)])
        check(f'patch order {h,w,p}',np.array_equal(got,ref))
        restored=got.reshape(h//p,w//p,p,p,3).transpose(0,2,1,3,4).reshape(h,w,3)
        check(f'patch reconstruction {h,w,p}',np.array_equal(a,restored))
    reject('patch tail rejected',lambda:patchify(np.zeros((5,4,3)),2))
    reject('patch channel rejected',lambda:patchify(np.zeros((4,4,1)),2))
    x=np.sin(2*np.pi*np.arange(48)/8);power,freq=stft_power(x,8000,16,8)
    oracle=[]
    for start in range(0,33,8):
        frame=x[start:start+16]*np.hanning(16)
        oracle.append([abs(sum(float(frame[n])*cmath.exp(-2j*math.pi*k*n/16) for n in range(16)))**2 for k in range(9)])
    check('STFT direct DFT oracle',np.allclose(power,oracle,rtol=1e-12,atol=1e-12))
    check('STFT bins',np.array_equal(freq,np.arange(9)*500))
    check('STFT count',power.shape==(5,9))
    p,_=stft_power(np.zeros(800));check('silence power finite zero',p.shape==(11,65) and np.array_equal(p,np.zeros_like(p)))
    reject('STFT stereo rejected',lambda:stft_power(np.zeros((800,2))))
    reject('STFT short rejected',lambda:stft_power(np.zeros(32)))
    reject('STFT NaN rejected',lambda:stft_power(np.full(800,np.nan)))
    reject('STFT hop rejected',lambda:stft_power(np.zeros(800),hop=256))
    h=hand();check('DFT hand',h['dft']['power']==[0.,4.,0.])
    splits=dataset();ids={k:{r['id'] for r in v} for k,v in splits.items()}
    check('group splits disjoint',all(not(ids[a]&ids[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]))
    check('split counts',{k:len(v) for k,v in splits.items()}==dict(train=96,validation=24,test=24))
    prep=fit(splits['train']);check('fit only train',set(prep['fit_ids'])==ids['train'])
    check('vocab size',len(prep['vocab'])==5)
    check('unicode NFC',words('ĐỎ DỌC')==['đỏ','dọc'])
    reject('unknown text rejected',lambda:bow('unknown',prep['vocab']))
    features=transform(splits['test'],prep)
    check('feature dimensions',{k:v.shape for k,v in features.items()}==dict(image=(24,768),audio=(24,715),text=(24,5)))
    check('features finite',all(np.isfinite(v).all() for v in features.values()))
    original=copy.deepcopy(prep);transform(splits['test'],prep);check('transform no refit',prep==original)
    bad=copy.deepcopy(prep);bad['spec']['sample_rate']=16000
    reject('spec mismatch rejected',lambda:transform(splits['test'],bad))
    bad=copy.deepcopy(prep);bad['std']['audio'][0]=0
    reject('normalizer zero rejected',lambda:transform(splits['test'],bad))
    for modality in ['image','audio']:
        check(modality+' rule baseline',np.array_equal(rule_predict(splits['test'],modality),[r['label'] for r in splits['test']]))
    # Multi-positive denominator: two matching +1 candidates and one orthogonal negative.
    a=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],dtype=torch.float64);labels=torch.tensor([0,0,1])
    actual=positive_set_loss(a,a,labels,tau=1.)
    expected=(2*math.log((2*math.e+1)/(2*math.e))+math.log((math.e+2)/math.e))/3
    check('multi-positive manual oracle',abs(float(actual)-expected)<1e-12)
    check('one-class loss zero',abs(float(positive_set_loss(a,a,torch.zeros(3,dtype=torch.long))))<1e-12)
    reject('loss shape rejected',lambda:positive_set_loss(a,a[:2],labels))
    reject('zero temperature rejected',lambda:positive_set_loss(a,a,labels,0))
    reject('infinite temperature rejected',lambda:positive_set_loss(a,a,labels,float('inf')))
    unique=torch.arange(3);scores=a@a.T/.15
    expected_ce=(torch.nn.functional.cross_entropy(scores,unique)+torch.nn.functional.cross_entropy(scores.T,unique))/2
    check('unique positives equal diagonal CE',torch.allclose(positive_set_loss(a,a,unique),expected_ce))
    # Direction symmetry, positive-set permutation invariance, and finite gradient.
    b=torch.tensor([[.8,.6],[.6,.8],[-1.,0.]],dtype=torch.float64)
    check('loss symmetric',torch.allclose(positive_set_loss(a,b,labels),positive_set_loss(b,a,labels)))
    perm=torch.tensor([2,0,1]);check('joint permutation invariant',torch.allclose(positive_set_loss(a,b,labels),positive_set_loss(a[perm],b[perm],labels[perm])))
    model=Align(len(prep['vocab']));x=tensor_features(splits['train'],prep);z=model(x)
    check('embedding normalized',all(torch.allclose(v.norm(dim=1),torch.ones(96),atol=1e-6) for v in z.values()))
    before=state_hash(model.state_dict());loss=total_loss(z,torch.tensor([r['label'] for r in splits['train']]));loss.backward()
    check('all encoders finite gradients',all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    for modality in ['image','audio','text']:
        check(modality+' receives nonzero gradient',sum(float(p.grad.norm()) for p in model.encoders[modality].parameters())>0)
    torch.optim.SGD(model.parameters(),lr=.01).step();check('actual parameter update',state_hash(model.state_dict())!=before)
    save_bundle(out/'test.pt',model,prep,fingerprint(splits));loaded,restored,_=load_bundle(out/'test.pt')
    with torch.no_grad():za=model(x);zb=loaded(x)
    check('bundle embedding exact',all(torch.equal(za[k],zb[k]) for k in z))
    check('bundle preprocess exact',restored==prep)
    reject('bundle expected spec rejected',lambda:load_bundle(out/'test.pt',dict(SPEC,sample_rate=16000)))
    b=torch.load(out/'test.pt',weights_only=True);b['preprocess']['mean']['image'][0]+=.1;torch.save(b,out/'bad.pt')
    reject('bundle preprocess digest rejected',lambda:load_bundle(out/'bad.pt'))
    b=torch.load(out/'test.pt',weights_only=True);b['state']['encoders.text.bias'][0]+=.1;torch.save(b,out/'bad.pt')
    reject('bundle state digest rejected',lambda:load_bundle(out/'bad.pt'))
    s1=np.array([[.9,.4]]);s2=np.array([[.2,.8]])
    check('fusion hand',np.allclose(late_fuse(s1,s2,.75),[[.725,.5]]))
    check('missing audio uses image',np.array_equal(late_fuse(s1,None),s1))
    check('missing image uses audio',np.array_equal(late_fuse(None,s2),s2))
    reject('all modalities missing rejected',lambda:late_fuse())
    reject('fusion invalid weight rejected',lambda:late_fuse(s1,s2,2))
    reject('missing branch NaN rejected',lambda:late_fuse(None,np.array([[np.nan]])))
    reject('metric shape rejected',lambda:top1(s1,[0],[0]))
    check('metric semantic positive',top1([[.1,.9,.8]],[1],[0,1,1])['correct']==1)
    sample=splits['test'][0];save_png(out/'sample.png',sample['image']);read=load_png(out/'sample.png')
    check('PNG roundtrip exact',np.array_equal(read,sample['image']))
    Image.fromarray(np.zeros((16,16),dtype=np.uint8)).save(out/'gray.png')
    reject('grayscale contract rejected',lambda:load_png(out/'gray.png'))
    write_wav(out/'sample.wav',sample['audio']);read=read_wav(out/'sample.wav')
    check('WAV roundtrip quantization',len(read)==800 and np.max(np.abs(read-sample['audio']))<=.5/32767+1e-12)
    with wave.open(str(out/'sample.wav'),'rb') as w:check('WAV metadata',w.getnchannels()==1 and w.getsampwidth()==2 and w.getframerate()==8000)
    reject('sample rate mismatch rejected',lambda:read_wav(out/'sample.wav',16000))
    reject('audio budget rejected',lambda:read_wav(out/'sample.wav',max_frames=400))
    check('context hypothetical',hand()['context']['total']==70 and hand()['context']['square_pairs']==4900)
    return dict(checks=len(names),passed=len(names),names=names)
