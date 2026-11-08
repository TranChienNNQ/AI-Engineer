import argparse,json,platform,tempfile
from pathlib import Path
from importlib.metadata import version
from corpus import documents,DEV,TEST,digest
from pipeline import Engine
from evaluation import run
from checks import run as check


def data_summary():return dict(documents=len(documents()),dev_cases=len(DEV),test_cases=len(TEST),corpus_hash=digest(documents()),eval_snapshot=digest(dict(dev=DEV,test=TEST)))

def hand():return dict(stage_success_product=.9*.95*.98,always_abstain_exact=.5,expected_review_minutes_per_100=100*(.6*2+.4*8),hypothetical_cost_per_request=.002+.3*.01,scope='illustrative probabilities/effort/cost only; not measured provider prices, token use or production savings')

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','experiment','check','serve'],default='all');p.add_argument('--out',type=Path);p.add_argument('--db',type=Path);p.add_argument('--port',type=int,default=8045);a=p.parse_args()
    if a.mode in ['all','experiment']:
        if a.out is None:p.error('--out required; use a fresh directory')
        if a.out.exists():p.error('output already exists')
    if a.mode=='serve':
        if a.db is None:p.error('--db required for local serving')
        import uvicorn
        from api import create_app
        engine=Engine(a.db)
        try:uvicorn.run(create_app(engine),host='127.0.0.1',port=a.port)
        finally:engine.close()
        return
    if a.mode=='hand':r=hand()
    elif a.mode=='data':r=data_summary()
    elif a.mode=='check':
        with tempfile.TemporaryDirectory() as directory:r=check(directory)
    elif a.mode=='experiment':r=run(a.out)
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        r=dict(environment=dict(python=platform.python_version(),fastapi=version('fastapi'),pydantic=version('pydantic'),httpx=version('httpx'),uvicorn=version('uvicorn'),platform=platform.machine(),device='cpu'),hand=hand(),data=data_summary(),experiment=run(a.out/'experiment'),checks=check(a.out/'checks'))
        (a.out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
