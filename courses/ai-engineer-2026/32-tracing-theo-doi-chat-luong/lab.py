"""Real local OTel tracing plus synthetic delayed-label monitoring, no external collector."""
import argparse
import asyncio
import json
import platform
from pathlib import Path
from importlib.metadata import version
from data import dataset_hash
from demos import trace_demo,trace_scenarios,incident_demo,hand_demo
from quality import quality_demo
from fixtures import evaluate


async def report():
    tracing=await trace_scenarios();quality=quality_demo()
    return dict(metadata=dict(python=platform.python_version(),opentelemetry_api=version('opentelemetry-api'),
                    opentelemetry_sdk=version('opentelemetry-sdk'),semantic_conventions=version('opentelemetry-semantic-conventions'),
                    fastapi=version('fastapi'),pydantic=version('pydantic'),httpx=version('httpx'),
                    dataset_hash=dataset_hash(),transport='Two ASGI apps in process, no TCP/collector/OTLP',
                    model='fixed numeric logistic forward; quality dataset is a separate synthetic replay',
                    timing='Real SDK timestamps and async waits; IDs/timings variable, not a performance benchmark'),
                tracing=tracing,quality=quality,evaluation=await evaluate(tracing,quality),hand=hand_demo())


async def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['trace','incident','quality','check','hand','all'],default='trace')
    p.add_argument('--drop-context',action='store_true');p.add_argument('--no-sampling',action='store_true')
    p.add_argument('--export-fail',action='store_true');p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.out and args.out.exists():raise FileExistsError('Choose a fresh output directory')
    if args.mode=='trace':result=await trace_demo(not args.drop_context,not args.no_sampling,args.export_fail)
    elif args.mode=='incident':result=await incident_demo()
    elif args.mode=='quality':result=quality_demo()
    elif args.mode=='check':result=await evaluate()
    elif args.mode=='hand':result=hand_demo()
    else:result=await report()
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False);print(text)
    if args.out:
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'report.json').write_text(text+'\n')


if __name__=='__main__':asyncio.run(main())
