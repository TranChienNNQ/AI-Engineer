import math
import torch
from torch import nn
from torch.nn import functional as F

MODALITIES=['image','audio','text']


class Align(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.encoders=nn.ModuleDict(dict(image=nn.Sequential(nn.Linear(768,32),nn.Tanh(),nn.Linear(32,12)),
            audio=nn.Sequential(nn.Linear(715,32),nn.Tanh(),nn.Linear(32,12)),text=nn.Linear(vocab_size,12)))
    def forward(self,x):
        return {k:F.normalize(self.encoders[k](x[k]),dim=-1,eps=1e-8) for k in MODALITIES}


def positive_set_loss(a,b,labels,tau=.15):
    if a.shape!=b.shape or a.ndim!=2 or len(a)!=len(labels) or not math.isfinite(tau) or tau<=0:raise ValueError('loss shape/temperature')
    logits=a@b.T/tau;positive=labels[:,None]==labels[None,:]
    # Positive-set probability, NOT mean log-prob per positive and not CLIP diagonal CE.
    def direction(scores,mask):return (torch.logsumexp(scores,dim=1)-torch.logsumexp(scores.masked_fill(~mask,-torch.inf),dim=1)).mean()
    return (direction(logits,positive)+direction(logits.T,positive.T))/2


def total_loss(z,labels):
    return sum(positive_set_loss(z[a],z[b],labels) for a,b in [('image','text'),('audio','text'),('image','audio')])/3
