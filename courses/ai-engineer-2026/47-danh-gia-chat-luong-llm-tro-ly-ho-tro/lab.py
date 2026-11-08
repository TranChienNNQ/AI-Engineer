import argparse,json,platform,tempfile
from pathlib import Path
from importlib.metadata import version
from dataset import cases,validate_split
from runner import run,analyze
from scoring import load_jsonl,wilson,agreement
from quality_checks import check

def hand():return dict(task_success_example=14/24,conditional_quality_example=8/14,zero_of_30_wilson95=wilson(0,30),judge_example=agreement([True,True,False,False],[True,False,True,False]),scope='arithmetic examples, not live model measurements')

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','experiment','check','score'],default='all');p.add_argument('--out',type=Path);p.add_argument('--records',type=Path);p.add_argument('--ratings',type=Path);a=p.parse_args()
    if a.mode in ['all','experiment','score']:
        if a.out is None:p.error('--out required; use a fresh directory')
        if a.out.exists():p.error('output already exists')
    if a.mode=='score':
        if a.records is None or a.ratings is None:p.error('--records and --ratings required')
        result=analyze(load_jsonl(a.records),load_jsonl(a.ratings))
        a.out.mkdir(parents=True,exist_ok=False);(a.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    elif a.mode=='hand':result=hand()
    elif a.mode=='data':result=dict(validation=validate_split(cases()),cases=cases())
    elif a.mode=='check':
        with tempfile.TemporaryDirectory() as d:result=check(Path(d))
    else:
        result=run(a.out)
        if a.mode=='all':
            result['checks']=check(a.out/'checks',result)
            result['environment']=dict(python=platform.python_version(),pydantic=version('pydantic'),httpx=version('httpx'),fastapi=version('fastapi'),device='cpu',machine=platform.machine())
            (a.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
