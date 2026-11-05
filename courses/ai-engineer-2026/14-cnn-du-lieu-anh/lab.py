"""Image files, hand convolution, and a tiny CNN. All images are generated locally."""
import argparse
import copy
import csv
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import numpy as np
from PIL import Image
import PIL
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
CLASSES=['vertical','horizontal']


def setup(seed=2026):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def out_size(size,kernel,stride=1,padding=0,dilation=1):
    return (size+2*padding-dilation*(kernel-1)-1)//stride+1


def conv_loop(image,kernel,stride=1,padding=0):
    """Independent NumPy reference: one channel, cross-correlation, dilation=1."""
    x=np.pad(np.asarray(image,dtype=np.float64),padding)
    k=np.asarray(kernel,dtype=np.float64);kh,kw=k.shape
    h=(x.shape[0]-kh)//stride+1;w=(x.shape[1]-kw)//stride+1
    result=np.zeros((h,w))
    for i in range(h):
        for j in range(w):result[i,j]=(x[i*stride:i*stride+kh,j*stride:j*stride+kw]*k).sum()
    return result


def hand_demo():
    image=np.array([[1,2,0],[0,1,3],[2,1,0]],dtype=np.float64)
    kernel=np.array([[1,0],[0,-1]],dtype=np.float64)
    x=torch.tensor(image).reshape(1,1,3,3)
    weight=torch.tensor(kernel).reshape(1,1,2,2).requires_grad_()
    bias=torch.tensor([.5],dtype=torch.float64,requires_grad=True)
    z=F.conv2d(x,weight,bias)
    reference=conv_loop(image,kernel)+.5
    np.testing.assert_array_equal(z.detach().numpy()[0,0],reference)
    torch.testing.assert_close(z,torch.tensor([[[[.5,-.5],[-.5,1.5]]]],dtype=torch.float64))
    z.sum().backward()
    torch.testing.assert_close(weight.grad,torch.tensor([[[[4.,6.],[4.,5.]]]],dtype=torch.float64))
    assert bias.grad.item()==4
    return {'convolution':reference.tolist(),'relu':F.relu(z).detach()[0,0].tolist(),
            'pool':F.max_pool2d(F.relu(z),2).item(),'weight_gradient':weight.grad[0,0].tolist(),
            'bias_gradient':bias.grad.item()}


def make_model():
    return nn.Sequential(nn.Conv2d(1,4,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(4,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Flatten(start_dim=1),nn.Linear(8*4*4,2)).to(device='cpu',dtype=torch.float32)


def shape_demo():
    model=make_model();x=torch.zeros(2,1,16,16);rows=[]
    for layer in model:
        x=layer(x);rows.append((type(layer).__name__,list(x.shape)))
    count=sum(p.numel() for p in model.parameters());assert count==594
    return {'shapes':rows,'parameters':count,'conv1_weight_shape':list(model[0].weight.shape)}


def make_corpus(root,seed=17):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    (root/'images').mkdir();rng=np.random.default_rng(seed)
    group_split={}
    for label in range(2):
        groups=rng.permutation(np.arange(label,64,2))
        for name,ids in [('train',groups[:20]),('validation',groups[20:26]),('test',groups[26:])]:
            for gid in ids:group_split[int(gid)]=name
    records=[]
    for gid in range(64):
        label=gid%2;position=int(rng.integers(2,13));brightness=float(rng.uniform(.65,1.))
        base=np.zeros((16,16),dtype=np.float64)
        if label==0:base[2:14,position:position+2]=brightness
        else:base[position:position+2,2:14]=brightness
        for view in range(2):
            image=np.clip(base+rng.normal(.04,.045,size=base.shape),0,1)
            pixels=np.rint(image*255).astype(np.uint8)
            name=f'images/g{gid:02d}-v{view}.png';Image.fromarray(pixels).save(root/name)
            records.append({'path':name,'label':label,'group':gid,'split':group_split[gid],
                            'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()})
    write_json(root/'manifest.json',{'classes':CLASSES,'seed':seed,'records':records})
    return records


def read_gray(path):
    with Image.open(path) as image:
        pixels=np.array(image.convert('L'),dtype=np.uint8,copy=True)
    if pixels.shape!=(16,16):raise ValueError('This model requires 16x16 grayscale input')
    return torch.from_numpy(pixels).to(torch.float32).unsqueeze(0)/255.


def fit_stats(root,records):
    pixels=torch.stack([read_gray(Path(root)/r['path']) for r in records if r['split']=='train'])
    mean=pixels.mean().item();std=pixels.std(correction=0).item()
    if std<=0:raise ValueError('Train standard deviation must be positive')
    return {'mean':mean,'std':std,'mode':'L','size':[16,16],'scale':255.}


class PNGDataset(Dataset):
    def __init__(self,root,records,split,stats,augment=False):
        if augment and split!='train':raise ValueError('Random augmentation is train-only')
        self.root=Path(root);self.rows=[r for r in records if r['split']==split]
        self.stats=stats;self.augment=augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];x=read_gray(self.root/row['path'])
        if self.augment and torch.rand(()).item()<.5:x=torch.flip(x,[-1])
        x=(x-self.stats['mean'])/self.stats['std']
        return x,torch.tensor(row['label'],dtype=torch.long)


def loader(dataset,batch_size=16,training=False,seed=2026):
    return DataLoader(dataset,batch_size=batch_size,shuffle=training,drop_last=False,num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(model,batches):
    model.eval();total=0.;count=0;confusion=torch.zeros((2,2),dtype=torch.long)
    for x,y in batches:
        logits=model(x)
        if logits.shape!=(len(y),2):raise ValueError('Expected N x 2 logits')
        total+=F.cross_entropy(logits,y,reduction='sum').item();count+=len(y)
        predicted=logits.argmax(1)
        confusion+=torch.bincount(2*y+predicted,minlength=4).reshape(2,2)
    if count==0:raise ValueError('Empty evaluation set')
    loss=total/count
    if not math.isfinite(loss):raise FloatingPointError('Non-finite evaluation loss')
    return {'loss':loss,'accuracy':confusion.diag().sum().item()/count,'n':count,'confusion':confusion.tolist()}


@dataclass(frozen=True)
class Config:
    epochs:int=30
    lr:float=.01
    batch_size:int=16
    seed:int=2026
    data_seed:int=17


def train(out,config=Config()):
    if config.epochs<=0 or config.lr<=0 or config.batch_size<=0:raise ValueError('Invalid configuration')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);setup(config.seed)
    records=make_corpus(out/'data',config.data_seed);stats=fit_stats(out/'data',records)
    data_hash=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    datasets={key:PNGDataset(out/'data',records,key,stats,augment=key=='train') for key in ['train','validation','test']}
    train_eval=PNGDataset(out/'data',records,'train',stats)
    model=make_model();optimizer=torch.optim.Adam(model.parameters(),lr=config.lr)
    write_json(out/'config.json',asdict(config));write_json(out/'preprocess.json',stats)
    write_json(out/'metadata.json',{'python':platform.python_version(),'torch':str(torch.__version__),
               'numpy':np.__version__,'pillow':PIL.__version__,'device':'cpu','dtype':'float32','threads':1,
               'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data_sha256':data_hash})
    best=math.inf;history=[];best_epoch=0
    for epoch in range(1,config.epochs+1):
        model.train();online=0.;n=0
        for x,y in loader(datasets['train'],config.batch_size,True,config.seed+epoch):
            optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x),y)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite train loss')
            loss.backward();optimizer.step();online+=loss.item()*len(y);n+=len(y)
        train_metric=evaluate(model,loader(train_eval,config.batch_size))
        val=evaluate(model,loader(datasets['validation'],config.batch_size))
        history.append({'epoch':epoch,'online_loss':online/n,'train_loss':train_metric['loss'],
                        'train_accuracy':train_metric['accuracy'],'val_loss':val['loss'],'val_accuracy':val['accuracy']})
        if val['loss']<best:
            best=val['loss'];best_epoch=epoch
            torch.save({'model':copy.deepcopy(model.state_dict()),'config':asdict(config),'preprocess':stats,
                        'classes':CLASSES,'epoch':epoch,'val_loss':best,'data_sha256':data_hash},out/'best.pt')
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    selected,checkpoint=load_best(out/'best.pt')
    val=evaluate(selected,loader(datasets['validation'],config.batch_size))
    test=evaluate(selected,loader(datasets['test'],config.batch_size))  # Only after best epoch selection.
    counts=torch.bincount(torch.tensor([r['label'] for r in records if r['split']=='train']),minlength=2)
    majority=int(counts.argmax())
    baseline=sum(r['label']==majority for r in records if r['split']=='validation')/len(datasets['validation'])
    # A concrete prediction roundtrip using the exact stored preprocessing contract.
    row=next(r for r in records if r['split']=='test')
    prediction=predict_file(out/'best.pt',out/'data'/row['path'])
    with torch.no_grad():
        x,_=datasets['test'][0];reference=selected(x.unsqueeze(0)).softmax(1)[0]
    torch.testing.assert_close(torch.tensor(prediction['probabilities']),reference,rtol=0,atol=0)
    report={'best_epoch':best_epoch,'best_val_loss':best,'validation':val,'test':test,
            'baseline_val_accuracy':baseline,'parameters':sum(p.numel() for p in model.parameters()),
            'train_n':len(datasets['train']),'preprocess':stats,'prediction_roundtrip':True}
    write_json(out/'report.json',report);return report


def load_best(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=make_model();model.load_state_dict(checkpoint['model']);model.eval();return model,checkpoint


def predict_file(checkpoint_path,image_path):
    model,checkpoint=load_best(checkpoint_path);stats=checkpoint['preprocess']
    x=(read_gray(image_path)-stats['mean'])/stats['std']
    with torch.no_grad():p=model(x.unsqueeze(0)).softmax(1)[0]
    label=int(p.argmax());return {'class_id':label,'class_name':checkpoint['classes'][label],'probabilities':p.tolist()}


def image_demo():
    # RGB memory layout example; reshape cannot replace axis permutation.
    rgb=np.zeros((2,3,3),dtype=np.uint8);rgb[0,1]=[255,128,0]
    chw=torch.from_numpy(rgb.copy()).permute(2,0,1).to(torch.float32)/255
    assert chw.shape==(3,2,3) and chw[0,0,1].item()==1
    vertical=torch.zeros(1,16,16);vertical[:,2:14,5:7]=1
    horizontal=torch.rot90(vertical,1,[-2,-1])
    assert vertical.sum()==horizontal.sum()
    assert vertical.sum(-2).var()>vertical.sum(-1).var()
    assert horizontal.sum(-2).var()<horizontal.sum(-1).var()
    return {'HWC':list(rgb.shape),'CHW':list(chw.shape),'pixel':chw[:,0,1].tolist(),
            'rotation90_changes_orientation':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['hand','shapes','images','train','all'],default='all')
    parser.add_argument('--out',type=Path);args=parser.parse_args();setup()
    for name,demo in [('hand',hand_demo),('shapes',shape_demo),('images',image_demo)]:
        if args.mode in (name,'all'):print(name,json.dumps(demo(),indent=2))
    if args.mode in ('train','all'):
        out=args.out if args.out else Path(tempfile.mkdtemp(prefix='lesson14-'))/'run'
        print('train',json.dumps(train(out),indent=2));print('Artifacts:',out.resolve())

if __name__=='__main__':main()
