"""Offline agent-loop/state/memory lab. Persist only to a user-selected synthetic SQLite file."""
import argparse
import json
import sqlite3
import platform
from pathlib import Path
from importlib.metadata import version
from contracts import DECISION,VERSION,digest
from fixtures import evaluate,memory_demo,hand_demo,scenario
from agent import Agent
from policy import POLICIES
from store import Store
from world import World


def report():
    w=World()
    return dict(metadata=dict(python=platform.python_version(),sqlite=sqlite3.sqlite_version,jsonschema=version('jsonschema'),device='CPU',
                              policy='scripted, no LLM',runtime_version=VERSION),
                dataset_hash=digest(dict(people=w.people,resources=w.resources)),contract_hash=digest(DECISION),
                hand=hand_demo(),memory=memory_demo(),evaluation=evaluate())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['hand','demo','memory','eval','all','start','resume'],default='demo')
    p.add_argument('--scenario',default='happy');p.add_argument('--out',type=Path)
    p.add_argument('--db',type=Path);p.add_argument('--run',default='run-28');p.add_argument('--actor',default='alice')
    p.add_argument('--intent',choices=['triage','retry'],default='triage');p.add_argument('--ticket',default='T-BLUE-1')
    p.add_argument('--steps',type=int,default=30);p.add_argument('--recover',action='store_true');args=p.parse_args()
    if args.out and args.out.exists():raise FileExistsError('Choose a fresh output directory')
    if args.mode in {'start','resume'}:
        if args.db is None:p.error('--db is required')
        if args.mode=='resume' and not args.db.is_file():p.error('Cannot resume a missing database')
        w=World();s=Store(args.db)
        try:
            if args.mode=='start':s.create(args.run,args.actor,w,ticket_id=args.ticket,intent=args.intent)
            state=s.load(args.run,args.actor,w)
            policy=POLICIES.get(state['model_id'])
            if policy is None:raise ValueError('Unknown policy needs an explicit compatible adapter')
            obj=Agent(s,w,args.run,args.actor,policy()).run(max_steps=args.steps,recover=args.recover)
        finally:s.close()
    elif args.mode=='hand':obj=hand_demo()
    elif args.mode=='demo':obj=scenario(args.scenario)
    elif args.mode=='memory':obj=memory_demo()
    else:obj=report()
    print(json.dumps(obj,ensure_ascii=False,indent=2))
    if args.out:
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/'report.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
