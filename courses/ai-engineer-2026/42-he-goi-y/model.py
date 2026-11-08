import torch
from torch import nn
from torch.nn import functional as F


class MF(nn.Module):
    def __init__(self,users=48,items=72,dim=12):
        super().__init__();self.user=nn.Embedding(users,dim);self.item=nn.Embedding(items,dim);self.bias=nn.Embedding(items,1)
        nn.init.normal_(self.user.weight,std=.05);nn.init.normal_(self.item.weight,std=.05);nn.init.zeros_(self.bias.weight)
    def forward(self,u,i):return (self.user(u)*self.item(i)).sum(dim=-1)+self.bias(i).squeeze(-1)


def bpr_loss(model,u,i,j,reg=.001):
    if u.shape!=i.shape or i.shape!=j.shape or u.ndim!=1 or len(u)==0 or reg<0:raise ValueError('BPR shape/regularization')
    delta=model(u,i)-model(u,j);pair=F.softplus(-delta).mean()
    # Mean per sampled triple; repeated users/items contribute repeatedly by design.
    penalty=(model.user(u).square().sum(dim=1)+model.item(i).square().sum(dim=1)+model.item(j).square().sum(dim=1)
             +model.bias(i).squeeze(-1).square()+model.bias(j).squeeze(-1).square()).mean()
    return pair+reg*penalty,pair
