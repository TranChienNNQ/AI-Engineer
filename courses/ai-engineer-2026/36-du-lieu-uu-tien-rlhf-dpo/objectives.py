"""Original sigmoid DPO: completion log-probability SUM, then mean over pairs."""
import math
import torch
from torch.nn import functional as F
from data import IGNORE


def positive(beta):
    if not math.isfinite(beta) or beta<=0:raise ValueError('finite beta > 0 required')


def sequence_logps(logits,labels):
    if logits.shape[:2]!=labels.shape or labels.ndim!=2:raise ValueError('shape')
    mask=labels.ne(IGNORE)
    if not mask.any(1).all():raise ValueError('every completion needs a token')
    safe=labels.masked_fill(~mask,0)
    token=logits.log_softmax(-1).gather(-1,safe.unsqueeze(-1)).squeeze(-1)
    return token.masked_fill(~mask,0).sum(-1)


def dpo_loss(pc,pr,rc,rr,beta=.5):
    positive(beta)
    if not(pc.shape==pr.shape==rc.shape==rr.shape) or pc.ndim!=1 or not pc.numel():raise ValueError('pair shape')
    if rc.requires_grad or rr.requires_grad:raise ValueError('reference must be detached/frozen')
    margin=(pc-rc)-(pr-rr)
    return -F.logsigmoid(beta*margin).mean(),margin


def hand():
    pc,pr,rc,rr=[torch.tensor([math.log(v)],dtype=torch.float64) for v in [.7,.2,.6,.3]]
    pc.requires_grad_();pr.requires_grad_();loss,m=dpo_loss(pc,pr,rc,rr,.5);loss.backward()
    p=torch.tensor([.75,.25],dtype=torch.float64);ref=torch.tensor([.5,.5],dtype=torch.float64)
    reward=torch.tensor([math.log(3),0],dtype=torch.float64);kl=(p*(p.log()-ref.log())).sum()
    return dict(margin=m.item(),scaled_margin=.5*m.item(),loss=loss.item(),d_pc=pc.grad.item(),d_pr=pr.grad.item(),
                rm_probability=.75,rm_loss=-math.log(.75),rm_gap=math.log(3),
                optimal_policy=p.tolist(),kl=kl.item(),expected_reward=(p*reward).sum().item(),objective=((p*reward).sum()-kl).item())
