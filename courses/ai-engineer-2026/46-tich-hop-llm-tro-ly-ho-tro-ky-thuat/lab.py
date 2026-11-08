import argparse,json,os,platform,tempfile
from pathlib import Path
from importlib.metadata import version
from bridge45 import digest,documents,request,source_hash
from contracts import identity
from fixtures import CASES
from evaluation import run
from checks import run as check


def data_summary():return dict(base45_source=source_hash(),corpus_hash=digest(documents()),fixture_cases=len(CASES),fixture_snapshot=digest(CASES),contracts=identity())
def hand():return dict(example_input_tokens=1000,example_output_tokens=200,hypothetical_input_per_million=.5,hypothetical_output_per_million=2.,hypothetical_cost=.0009,retry_probability=.1,expected_attempts_max_two=1.1,one_timeout_usage='unknown; never assume zero',scope='assumed prices/tokens, not provider pricing or a measured model run')


def live(args):
    from llm_adapter import ResponsesAdapter
    from support import SupportEngine
    key=os.environ.get('OPENAI_API_KEY','');model=args.model or os.environ.get('LLM_MODEL','')
    if not key or not model:raise SystemExit('OPENAI_API_KEY and --model (or LLM_MODEL) are required; no request sent')
    if not args.message:raise SystemExit('--message is required for live mode; no request sent')
    args.out.mkdir(parents=True,exist_ok=False)
    adapter=ResponsesAdapter(model,key,mode='live');e=SupportEngine(args.out/'project.sqlite',adapter,policy=args.policy)
    try:result=e.ask('demo-agent-a',request(args.message,action=args.action,key='live-run',service=args.service,ticket_id=args.ticket_id,expected_version=args.expected_version))
    finally:e.close()
    (args.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','experiment','check','live'],default='all');p.add_argument('--out',type=Path);p.add_argument('--model');p.add_argument('--message');p.add_argument('--service',choices=['gateway','upload','identity','database'],default='gateway');p.add_argument('--action',choices=['ask','create','note','resolve'],default='ask');p.add_argument('--ticket-id',type=int);p.add_argument('--expected-version',type=int);p.add_argument('--policy',choices=['exact_quote','citation_only'],default='exact_quote');a=p.parse_args()
    if a.mode in ['all','experiment','live']:
        if a.out is None:p.error('--out required; choose a fresh output directory')
        if a.out.exists():p.error('output already exists')
    if a.mode=='live':live(a);return
    if a.mode=='hand':r=hand()
    elif a.mode=='data':r=data_summary()
    elif a.mode=='check':
        with tempfile.TemporaryDirectory() as directory:r=check(directory)
    elif a.mode=='experiment':r=run(a.out)
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        r=dict(environment=dict(python=platform.python_version(),pydantic=version('pydantic'),httpx=version('httpx'),fastapi=version('fastapi'),platform=platform.machine(),device='cpu'),hand=hand(),data=data_summary(),experiment=run(a.out/'experiment'),checks=check(a.out/'checks'))
        (a.out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
