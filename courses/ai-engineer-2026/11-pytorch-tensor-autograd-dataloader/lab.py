"""Lesson 11: CPU PyTorch, hand gradients and DataLoader; all data synthetic."""
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
DTYPE=torch.float64

class TinyDataset(Dataset):
    """Five constructed regression samples; one feature and one target per row."""
    def __init__(self):
        self.X=torch.arange(5,dtype=DTYPE).reshape(-1,1)
        self.y=2*self.X+1
    def __len__(self): return len(self.X)
    def __getitem__(self,index): return self.X[index],self.y[index]

def make_loader(shuffle=False,drop_last=False,seed=2026):
    return DataLoader(TinyDataset(),batch_size=2,shuffle=shuffle,drop_last=drop_last,
                      num_workers=0,generator=torch.Generator().manual_seed(seed))

def tensor_demo():
    a=torch.tensor([[1.,2.],[3.,4.]],dtype=DTYPE)
    print('shape/dtype/device:',a.shape,a.dtype,a.device)
    print('elementwise:',a*a,'matmul:',a@a,sep='\n')
    source=np.array([1.,2.]);shared=torch.from_numpy(source);copied=torch.tensor(source)
    source[0]=9
    print('NumPy shared:',shared,'copied:',copied)
    print('one-row flatten:',a[:1].shape,'target column:',a[:,0:1].shape)

def scalar_graph():
    x=torch.tensor(2.,dtype=DTYPE)
    w=torch.tensor(.5,dtype=DTYPE,requires_grad=True)
    b=torch.tensor(1.,dtype=DTYPE,requires_grad=True)
    a=w*x+b
    a.retain_grad()
    loss=a.square()+3*a
    loss.backward()
    return w,b,a,loss

def autograd_demo():
    w,b,a,loss=scalar_graph()
    print('L,a.grad,w.grad,b.grad:',loss.item(),a.grad.item(),w.grad.item(),b.grad.item())
    # A new forward graph; .grad buffers on leaves persist until cleared.
    a2=w*2+b
    (a2.square()+3*a2).backward()
    print('After second forward/backward, accumulated:',w.grad.item(),b.grad.item())
    w.grad=None;b.grad=None
    a3=w*2+b
    (a3.square()+3*a3).backward()
    print('After clearing:',w.grad.item(),b.grad.item())
    detached=a.detach()
    print('detach requires_grad:',detached.requires_grad)
    print('independent snapshot:',a.detach().clone())

def hand_model():
    model=nn.Sequential(nn.Linear(2,2),nn.ReLU(),nn.Linear(2,1)).to(dtype=DTYPE,device='cpu')
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.,.5],[-1.,1.]],dtype=DTYPE))
        model[0].bias.copy_(torch.tensor([-1.,0.],dtype=DTYPE))
        model[2].weight.copy_(torch.tensor([[2.,-1.]],dtype=DTYPE))
        model[2].bias.copy_(torch.tensor([.5],dtype=DTYPE))
    return model

def hand_backward():
    X=torch.tensor([[1.,2.],[-1.,1.]],dtype=DTYPE)
    y=torch.tensor([[1.],[0.]],dtype=DTYPE)
    model=hand_model();logits=model(X)
    loss=nn.BCEWithLogitsLoss()(logits,y)
    loss.backward()
    q=1/(1+np.exp(1.5))
    # Expected gradients in PYTORCH's (out,in) weight layout.
    expected=[torch.tensor([[-q,-2*q],[q,q/2]],dtype=DTYPE),
              torch.tensor([-q,0.],dtype=DTYPE),
              torch.tensor([[-q/2,q/2]],dtype=DTYPE),torch.zeros(1,dtype=DTYPE)]
    for param,answer in zip(model.parameters(),expected):
        torch.testing.assert_close(param.grad,answer,atol=1e-14,rtol=1e-12)
    old=loss.item()
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    optimizer.step()
    with torch.no_grad(): new=nn.BCEWithLogitsLoss()(model(X),y).item()
    return model,logits.detach(),expected,old,new

def hand_demo():
    model,logits,expected,old,new=hand_backward()
    print('logits:',logits,'BCE old/new:',old,new)
    for name,param in model.named_parameters(): print(name,'gradient:',param.grad)
    print('All four parameter gradients match lesson 10, with weight transposes.')

def loader_demo():
    for drop in (False,True):
        batches=list(make_loader(drop_last=drop))
        print('drop_last:',drop,'batch sizes:',[len(x) for x,y in batches])
        for X,y in batches: print('X/y:',X[:,0].tolist(),y[:,0].tolist(),'shapes:',tuple(X.shape),tuple(y.shape))
    # Evaluation of a FIXED zero predictor: no updates while aggregating losses.
    means=[float(y.square().mean()) for X,y in make_loader()]
    sizes=[len(X) for X,y in make_loader()]
    weighted=sum(a*b for a,b in zip(means,sizes))/sum(sizes)
    print('Zero-predictor batch MSE:',means,'naive mean:',sum(means)/len(means),'sample mean:',weighted)
    return means,sizes,weighted

def train_toy(epochs=100):
    torch.manual_seed(2026)
    model=nn.Linear(1,1).to(dtype=DTYPE)
    with torch.no_grad(): model.weight.zero_();model.bias.zero_()
    optimizer=torch.optim.SGD(model.parameters(),lr=.05)
    criterion=nn.MSELoss();history=[]
    for epoch in range(epochs+1):
        model.eval()
        with torch.no_grad():
            data=TinyDataset();history.append(criterion(model(data.X),data.y).item())
        if epoch==epochs: break
        model.train()
        for X,y in make_loader(shuffle=True,seed=2026+epoch):
            optimizer.zero_grad(set_to_none=True)
            prediction=model(X)
            if prediction.shape!=y.shape: raise ValueError('Prediction/target shapes must match.')
            loss=criterion(prediction,y)
            loss.backward()
            optimizer.step()
    return model,np.array(history)

def train_demo():
    model,history=train_toy()
    print('Train MSE before/after:',history[0],history[-1])
    print('weight/bias:',model.weight.item(),model.bias.item())
    print('100 epochs, 3 batches per epoch = 300 updates; synthetic TRAIN only.')

def modes_demo():
    model=hand_model();X=torch.ones((1,2),dtype=DTYPE)
    model.eval();print('eval only tracks graph:',model(X).requires_grad)
    with torch.no_grad(): print('no_grad tracks graph:',model(X).requires_grad)
    print('Normal graph after leaving no_grad:',model(X).requires_grad)

def main():
    torch.set_num_threads(1)
    print('PyTorch:',torch.__version__,'; CPU; float64 for numeric comparisons')
    demos={'tensor':tensor_demo,'autograd':autograd_demo,'hand':hand_demo,'loader':loader_demo,
           'train':train_demo,'modes':modes_demo}
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=[*demos,'all'],default='all');args=parser.parse_args()
    for name,demo in demos.items():
        if args.mode in (name,'all'): print('\n---',name,'---');demo()
if __name__=='__main__':main()
