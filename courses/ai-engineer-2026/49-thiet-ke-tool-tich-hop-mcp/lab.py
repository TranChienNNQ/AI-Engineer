import argparse,json,platform,tempfile
from pathlib import Path
from importlib.metadata import version
from experiment49 import run,CASES
from quality_checks import check
from contracts49 import MAP

def hand():return dict(model_turns_triage=4,mcp_tool_calls=3,catalog_list_calls=4,audit_reads=1,host_counted_operations=8,discovery_scope='SDK negotiation not included in ops; no latency/billing claim')
def data():return dict(cases=len(CASES),routes=MAP,transports=['memory','stdio'],fixture_model=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','experiment','check'],default='all');p.add_argument('--out',type=Path);a=p.parse_args()
 if a.mode in ['all','experiment']:
  if a.out is None:p.error('--out required; choose a fresh directory')
  if a.out.exists():p.error('output already exists')
 if a.mode=='hand':result=hand()
 elif a.mode=='data':result=data()
 elif a.mode=='check':
  with tempfile.TemporaryDirectory() as d:result=check(d)
 else:
  result=run(a.out)
  if a.mode=='all':
   result['checks']=check(a.out/'checks',result)
   result['environment']={n:version(n) for n in ['mcp','mcp-types','pydantic','anyio','jsonschema','httpx','httpx2','fastapi']}
   result['environment'].update(python=platform.python_version(),machine=platform.machine(),device='cpu')
   (a.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
