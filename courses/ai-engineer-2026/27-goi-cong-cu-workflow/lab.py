"""Run local tool/workflow demos. No provider key, network, actual ticket or arbitrary shell."""
import argparse
import asyncio
import json
from importlib.metadata import version
from pathlib import Path
import platform
from backend import Backend
from contracts import digest,TURN_SCHEMA
from fixtures import evaluate,scenario,protocol_demo
from parallel import read_fanout,hand_schedule


def report():
    b=Backend()
    dataset=dict(people=b.people,tickets=b.tickets,kb=[(list(k),v) for k,v in sorted(b.kb.items())],health=[(list(k),v) for k,v in sorted(b.health.items())],latency=b.latency)
    return dict(fixture_version='ticket-workflow-v1',contract_hash=digest(TURN_SCHEMA),dataset_hash=digest(dataset),metadata=dict(python=platform.python_version(),jsonschema=version('jsonschema'),device='CPU',
                              model='scripted template, no LLM',timing='workflow modeled ms; fanout actual scaled sleep'),
                hand=hand_schedule(),protocol=protocol_demo(),evaluation=evaluate())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['hand','workflow','protocol','parallel','eval','all'],default='workflow')
    parser.add_argument('--scenario',default='happy',help='See fixtures.py for all scenario names')
    parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.out and args.out.exists():raise FileExistsError('Choose a fresh output directory')
    if args.mode=='hand':obj=hand_schedule()
    elif args.mode=='workflow':obj=scenario(args.scenario)
    elif args.mode=='protocol':obj=protocol_demo()
    elif args.mode=='parallel':obj=asyncio.run(read_fanout(Backend()))
    else:
        obj=report()
        if args.mode=='all':obj['parallel']=asyncio.run(read_fanout(Backend()))
    print(json.dumps(obj,ensure_ascii=False,indent=2))
    if args.out:
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/'report.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
