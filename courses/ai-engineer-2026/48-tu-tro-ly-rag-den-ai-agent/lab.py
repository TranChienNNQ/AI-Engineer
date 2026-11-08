import argparse,json,os,platform,tempfile
from pathlib import Path
from importlib.metadata import version
from fixtures import CASES,task
from agent_contract import TOOLS
from experiment import run
from checks import check

def hand():return dict(triage_tool_calls=3,triage_model_turns=4,retry_tool_calls=2,retry_model_turns=3,example_attempts_with_one_retry=5,hypothetical_turn_cost=.0002,hypothetical_four_turn_cost=.0008,scope='arithmetic examples, not provider pricing or measured latency')
def data():return dict(cases=len(CASES),tools=[t['name'] for t in TOOLS],profiles=CASES,scope='scripted observation-dependent decisions, not LLM samples')

def live(args):
    from tools import World
    from runtime import Agent
    from provider import ToolAdapter
    key=os.environ.get('OPENAI_API_KEY','');model=args.model or os.environ.get('LLM_MODEL','')
    if not key or not model:raise SystemExit('OPENAI_API_KEY and --model or LLM_MODEL required; no request sent')
    if not args.question:raise SystemExit('--question required; no request sent')
    raw=task(goal=args.goal,question=args.question)
    from agent_contract import Task
    Task.model_validate(raw)
    args.out.mkdir(parents=True,exist_ok=False)
    w=World(args.out/'world.sqlite');adapter=ToolAdapter(model,key,mode='live');a=Agent(w,adapter)
    try:
        result=a.run('demo-agent-a',raw);manifest=a.manifest
    finally:a.close()
    result['manifest']=manifest
    (args.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','experiment','check','live'],default='all');p.add_argument('--out',type=Path);p.add_argument('--model');p.add_argument('--question');p.add_argument('--goal',choices=['retry','triage'],default='triage');a=p.parse_args()
    if a.mode in ['all','experiment','live']:
        if a.out is None:p.error('--out required; choose a fresh directory')
        if a.out.exists():p.error('output already exists')
    if a.mode=='live':live(a);return
    if a.mode=='hand':result=hand()
    elif a.mode=='data':result=data()
    elif a.mode=='check':
        with tempfile.TemporaryDirectory() as d:result=check(d)
    else:
        result=run(a.out)
        if a.mode=='all':
            result['checks']=check(a.out/'checks',result);result['environment']=dict(python=platform.python_version(),pydantic=version('pydantic'),httpx=version('httpx'),fastapi=version('fastapi'),machine=platform.machine(),device='cpu')
            (a.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
