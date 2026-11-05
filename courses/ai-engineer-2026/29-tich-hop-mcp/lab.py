"""MCP SDK lab: real stdio processes, in-memory SDK, and Streamable HTTP via in-process ASGI."""
import argparse
import asyncio
import json
from pathlib import Path
import platform
from importlib.metadata import version
from agent import demo
from domain import Backend,digest
from fixtures import evaluate,hand_demo
from wire import wire_demo


async def report():
    b=Backend()
    return dict(metadata=dict(python=platform.python_version(),mcp=version('mcp'),mcp_types=version('mcp-types'),
                              pydantic=version('pydantic'),jsonschema=version('jsonschema'),httpx2=version('httpx2'),
                              protocol='2026-07-28',device='CPU',model='scripted host, no LLM'),
                dataset_hash=digest(dict(tickets=b.tickets,docs=b.docs)),demo=await demo('stdio'),
                wire=await wire_demo(),evaluation=await evaluate(),hand=hand_demo())


async def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['demo','wire','eval','hand','all'],default='demo')
    p.add_argument('--transport',choices=['stdio','memory','http-asgi'],default='stdio')
    p.add_argument('--workspace',choices=['blue','green'],default='blue');p.add_argument('--legacy',action='store_true')
    p.add_argument('--out',type=Path);args=p.parse_args()
    if args.out and args.out.exists():raise FileExistsError('Choose a fresh output directory')
    if args.mode=='demo':obj=await demo(args.transport,args.workspace,'legacy' if args.legacy else 'auto')
    elif args.mode=='wire':obj=await wire_demo()
    elif args.mode=='eval':obj=await evaluate()
    elif args.mode=='hand':obj=hand_demo()
    else:obj=await report()
    print(json.dumps(obj,ensure_ascii=False,indent=2))
    if args.out:
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/'report.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))

if __name__=='__main__':asyncio.run(main())
