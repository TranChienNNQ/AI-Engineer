"""SFT/LoRA/QLoRA foundations: local CPU training, mechanism checks and optional library smoke."""
import argparse
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from data import records,Tokenizer
from demos import hand,mask_demo,quant_demo,memory
from fixtures import evaluate
from training import train
from gpu_recipe import preflight


def report(out):
    experiment=train(out)
    return dict(metadata=dict(python=platform.python_version(),torch=torch.__version__,device='CPU',threads=1,
                              cuda_available=torch.cuda.is_available(),gpu_qlora_executed=False),
                experiment=experiment,hand=hand(),mask=mask_demo(),quantization=quant_demo(),memory=memory(),
                evaluation=evaluate(),preflight=preflight())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['hand','mask','quant','check','train','library','all'],default='check')
    p.add_argument('--out',type=Path);a=p.parse_args()
    if a.out and a.out.exists():raise FileExistsError('Choose a fresh output directory')
    if a.mode in {'train','all'}:
        if a.out:r=report(a.out) if a.mode=='all' else train(a.out)
        else:
            with TemporaryDirectory(prefix='lesson35-') as tmp:r=report(Path(tmp)/'run') if a.mode=='all' else train(Path(tmp)/'run')
    elif a.mode=='hand':r=dict(lora=hand(),memory=memory())
    elif a.mode=='mask':r=mask_demo()
    elif a.mode=='quant':r=quant_demo()
    elif a.mode=='library':
        from library_demo import run
        r=run()
    else:r=evaluate()
    text=json.dumps(r,indent=2,allow_nan=False);print(text)
    if a.out:
        a.out.mkdir(parents=True,exist_ok=True);(a.out/'report.json').write_text(text+'\n')


if __name__=='__main__':main()
