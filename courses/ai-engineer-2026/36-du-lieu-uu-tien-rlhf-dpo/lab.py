"""Run from this directory; reports/checkpoints go to a fresh directory outside source."""
import argparse
import json
import platform
from pathlib import Path
import torch
from objectives import hand
from checks import run as check
from bandit import run as bandit
from training import experiment
from data import Tokenizer,records,example


def mask():
    tok=Tokenizer(records()['train']);x,y=example(tok,'route timeout','NET CHECK')
    return dict(input=[tok.words[i] for i in x],target=['IGNORE' if i==-100 else tok.words[i] for i in y])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['all','hand','mask','bandit','train','check'],default='all')
    parser.add_argument('--out',type=Path);args=parser.parse_args();torch.set_num_threads(1)
    if args.mode in ['all','train']:
        if args.out is None:parser.error('--out required for all/train; choose a fresh directory')
        args.out.mkdir(parents=True,exist_ok=False)
    funcs=dict(hand=hand,mask=mask,bandit=bandit,check=check,train=lambda:experiment(args.out))
    if args.mode=='all':
        result=dict(environment=dict(python=platform.python_version(),torch=torch.__version__,device='cpu'),
                    hand=hand(),mask=mask(),bandit=bandit(),checks=check(),experiment=experiment(args.out))
    else:result=funcs[args.mode]()
    if args.mode in ['all','train']:(args.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
