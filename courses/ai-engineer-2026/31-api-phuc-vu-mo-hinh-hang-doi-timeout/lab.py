"""Model serving lab: actual async queues/timers and ASGI; optional TCP loopback."""
import argparse
import asyncio
import json
import platform
from pathlib import Path
from importlib.metadata import version
from model import ARTIFACT_HASH
from demos import happy_demo,scenarios,hand_demo
from fixtures import evaluate


async def report():
    results=await scenarios()
    return dict(metadata=dict(python=platform.python_version(),fastapi=version('fastapi'),
                    pydantic=version('pydantic'),starlette=version('starlette'),httpx=version('httpx'),
                    uvicorn=version('uvicorn'),anyio=version('anyio'),artifact_hash=ARTIFACT_HASH,
                    model='fixed-weight numeric logistic model, not LLM or trained quality baseline',
                    transport='HTTP-ASGI in process; socket CLI tested separately',
                    timing='real event-loop deadlines, artificial cooperative delay; no throughput benchmark'),
                scenarios=results,evaluation=await evaluate(results),hand=hand_demo())


async def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['demo','scenarios','check','hand','socket','all'],default='demo')
    p.add_argument('--out',type=Path);args=p.parse_args()
    if args.out and args.out.exists():raise FileExistsError('Choose a fresh output directory')
    if args.mode=='demo':result=await happy_demo()
    elif args.mode=='scenarios':result=await scenarios()
    elif args.mode=='check':result=await evaluate()
    elif args.mode=='hand':result=hand_demo()
    elif args.mode=='socket':
        from socket_demo import socket_demo
        result=await socket_demo()
    else:result=await report()
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False);print(text)
    if args.out:
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'report.json').write_text(text+'\n')


if __name__=='__main__':asyncio.run(main())
