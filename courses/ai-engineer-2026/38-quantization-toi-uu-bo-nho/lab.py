import argparse
import json
import platform
from pathlib import Path
import torch
from arithmetic import hand,plan
from attention import run as attention
from checks import run as check
from training import experiment


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','memory','attention','check','train'],default='all');p.add_argument('--out',type=Path);a=p.parse_args();torch.set_num_threads(1)
    if a.mode in ['all','train']:
        if a.out is None:p.error('--out required; choose a fresh directory')
        a.out.mkdir(parents=True,exist_ok=False)
    f=dict(hand=hand,memory=plan,attention=attention,check=check,train=lambda:experiment(a.out))
    if a.mode=='all':r=dict(environment=dict(python=platform.python_version(),torch=torch.__version__,platform=platform.machine(),device='cpu',cuda=torch.cuda.is_available()),
                          hand=hand(),attention=attention(),checks=check(),experiment=experiment(a.out))
    else:r=f[a.mode]()
    if a.mode in ['all','train']:(a.out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
