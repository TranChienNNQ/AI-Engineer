import argparse,json,tempfile,platform
from pathlib import Path
from importlib.metadata import version
from experiment50 import run,CASES
from plan_checks import check

def hand():return dict(expected_value_example=.6*10-2,repair_calls_example=4,baseline_calls_example=2,additional_calls=2,scope='assumed information value/cost and scripted call counts; no calibrated probabilities or measured money')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','experiment','check'],default='all');p.add_argument('--out',type=Path);a=p.parse_args()
 if a.mode in ['all','experiment']:
  if a.out is None:p.error('--out required; choose a fresh directory')
  if a.out.exists():p.error('output already exists')
 if a.mode=='hand':result=hand()
 elif a.mode=='data':result=dict(cases=len(CASES),profiles=CASES,scope='scripted planning proposals; synthetic ticket setup; real MCP')
 elif a.mode=='check':
  with tempfile.TemporaryDirectory() as d:result=check(d)
 else:
  result=run(a.out)
  if a.mode=='all':
   result['checks']=check(a.out/'checks',result);result['environment']={n:version(n) for n in ['mcp','mcp-types','pydantic','anyio','jsonschema','httpx','fastapi']}
   result['environment'].update(python=platform.python_version(),machine=platform.machine(),device='cpu')
   (a.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
