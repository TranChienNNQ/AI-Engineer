import argparse
import json
import platform
from pathlib import Path
import torch
from model import make_model
from generation import generate
from arithmetic import hand
from scheduling import demo as schedule
from checks import run as check
from benchmark import run as benchmark


def cache_demo():
    m=make_model();prompts=[[1,4,6],[1,7,8,9,10],[1,3]]
    a=generate(m,prompts,8,True);b=generate(m,prompts,8,False)
    if a['outputs']!=b['outputs']:raise AssertionError('cache mismatch')
    return dict(prompts=prompts,cached=a,full=b,outputs_identical=True,model_parameters=sum(p.numel() for p in m.parameters()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','cache','schedule','check','bench'],default='all')
    p.add_argument('--out',type=Path);p.add_argument('--repeats',type=int,default=7);a=p.parse_args();torch.set_num_threads(1)
    if a.mode in ['all','bench']:
        if a.out is None:p.error('--out required for all/bench; choose a fresh directory')
        a.out.mkdir(parents=True,exist_ok=False)
    f=dict(hand=hand,cache=cache_demo,schedule=schedule,check=check,bench=lambda:benchmark(a.repeats))
    if a.mode=='all':r=dict(environment=dict(python=platform.python_version(),torch=torch.__version__,platform=platform.machine(),cuda=torch.cuda.is_available()),
                          hand=hand(),cache=cache_demo(),schedule=schedule(),checks=check(),benchmark=benchmark(a.repeats))
    else:r=f[a.mode]()
    if a.mode in ['all','bench']:(a.out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
