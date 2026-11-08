"""Measurement lesson: deterministic model, hypothetical costs and bounded local HTTP load."""
import argparse
import asyncio
import hashlib
import json
import platform
from pathlib import Path
from importlib.metadata import version
from measurements import distribution,streaming_example
from simulation import scenarios
from costs import self_hosted_example
from fixtures import evaluate,validate_tcp
from benchmark import benchmark


def hand():
    return dict(latencies_ms=[10,20,30,40,500],distribution_ms=distribution([10,20,30,40,500]),
                little_law=dict(arrival_per_second=5,mean_seconds=.2,mean_in_system=1),
                parallel=dict(branches_ms=[40,60],reduce_ms=10,critical_ms=70,total_work_ms=110),
                streaming=streaming_example())


def source_hash():
    root=Path(__file__).resolve().parent
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
    files['requirements.txt']=hashlib.sha256((root/'requirements.txt').read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()


async def report(repeats=2):
    tcp=await benchmark(repeats)
    return dict(metadata=dict(python=platform.python_version(),system=platform.system(),machine=platform.machine(),
                              versions={name:version(name) for name in ['fastapi','pydantic','httpx','uvicorn','starlette','anyio']},
                              source_hash=source_hash(),clock='perf_counter, monotonic elapsed durations',
                              prices='hypothetical, not current provider prices'),
                simulation=scenarios(),cost=self_hosted_example(),hand=hand(),tcp=tcp,
                evaluation=await evaluate(),tcp_validation=validate_tcp(tcp))


def stable_facts(r):
    # Wall times, overload/drop counts and completion rates are actual measurements, not reproducible constants.
    return {k:r[k] for k in ['metadata','simulation','cost','hand','evaluation']} | {
        'tcp_configs':[(x['profile'],x['repetition'],x['config']) for x in r['tcp']['runs']]}


async def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['hand','simulate','cost','tcp','check','all'],default='simulate')
    p.add_argument('--repeats',type=int,choices=[1,2,3],default=2)
    p.add_argument('--out',type=Path);args=p.parse_args()
    if args.out and args.out.exists():raise FileExistsError('Choose a fresh output directory')
    if args.mode=='hand':r=hand()
    elif args.mode=='simulate':r=scenarios()
    elif args.mode=='cost':r=self_hosted_example()
    elif args.mode=='tcp':
        tcp=await benchmark(args.repeats);r=dict(tcp=tcp,validation=validate_tcp(tcp))
    elif args.mode=='check':r=await evaluate()
    else:r=await report(args.repeats)
    text=json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False);print(text)
    if args.out:
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'report.json').write_text(text+'\n')


if __name__=='__main__':asyncio.run(main())
