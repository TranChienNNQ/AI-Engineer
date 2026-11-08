"""Multi-step evaluation and bounded action lab. Standard library only; no LLM."""
import argparse
import json
import platform
from pathlib import Path
from dataclasses import asdict
from domain import cases,manifest,digest
from gateway import Limits
from evaluation import trial,suite,compare,hand_demo
from fixtures import checks


def report():
    return dict(metadata=dict(python=platform.python_version(),version='30.0',model='scripted policies, no LLM',
                              clock='modeled integer ms, no real deadline test',cost='synthetic units, no currency',
                              grader='deterministic domain facts + host-controlled audit',default_limits=asdict(Limits())),
                manifest_hash=digest(manifest()),demo=trial(cases()[0]),
                evaluation=suite('all'),comparison=compare('all'),checks=checks(),hand=hand_demo())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['demo','eval','compare','check','hand','all'],default='demo')
    p.add_argument('--split',choices=['dev','holdout','all'],default='dev')
    p.add_argument('--policy',choices=['adaptive','reckless','refuse'],default='adaptive')
    p.add_argument('--boundary',choices=['strict','schema_only'],default='strict')
    p.add_argument('--profile',type=int,choices=range(4),default=0)
    p.add_argument('--task',default='D01');p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.out and args.out.exists():raise FileExistsError('Choose a fresh output directory')
    if args.mode=='demo':
        chosen=next((c for c in cases() if c.case_id==args.task),None)
        if chosen is None:p.error('unknown task ID')
        result=trial(chosen,args.policy,args.profile,args.boundary)
    elif args.mode=='eval':result=suite(args.split,args.policy,args.boundary)
    elif args.mode=='compare':result=compare(args.split)
    elif args.mode=='check':result=checks()
    elif args.mode=='hand':result=hand_demo()
    else:result=report()
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False);print(text)
    if args.out:
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'report.json').write_text(text+'\n')


if __name__=='__main__':main()
