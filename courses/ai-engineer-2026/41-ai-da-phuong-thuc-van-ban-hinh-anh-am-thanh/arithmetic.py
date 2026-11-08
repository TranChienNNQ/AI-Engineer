import math
import numpy as np


def hand():
    scores=np.array([[1.,0.],[.6,.8]])/.5
    losses=np.log(np.exp(scores).sum(axis=1))-np.diag(scores)
    # Symmetric diagonal contrastive example only; lab training handles repeated class positives.
    cols=np.log(np.exp(scores).sum(axis=0))-np.diag(scores)
    x=np.array([0.,1.,0.,-1.]);spectrum=np.fft.rfft(x)
    return dict(patches=dict(image_shape=[4,4,3],patch=2,count=4,flattened_dim=12),
        audio=dict(sample_rate=8000,samples=800,duration_s=.1,frame=128,hop=64,frames=11,bins=65,bin_hz=62.5,discarded_tail_samples=32),
        dft=dict(input=x.tolist(),real=spectrum.real.tolist(),imag=spectrum.imag.tolist(),power=(np.abs(spectrum)**2).tolist()),
        contrastive=dict(cosines=[[1.,0.],[.6,.8]],tau=.5,row_loss=losses.tolist(),symmetric_loss=float((losses.mean()+cols.mean())/2)),
        fusion=dict(image_scores=[.9,.4],audio_scores=[.2,.8],image_weight=.75,scores=[.725,.5]),
        context=dict(text=40,image_patches=16,audio_frames=11,special=3,total=70,square_pairs=4900,
            note='hypothetical concatenation counts before any pooling/downsampling; not API billing'))
