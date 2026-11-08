import argparse,json,tempfile,platform,sqlite3
from pathlib import Path
from importlib.metadata import version
from experiment51 import run,CASES
from checks51 import check

def hand():return dict(reserved_before_crash=2,acknowledged_before_crash=2,committed_observations_before_crash=1,reservations_after_recovery=5,acknowledgements_after_recovery=5,committed_observations_after_recovery=4,scope='after_response synthetic read-only case; reservations are not physical executions')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','experiment','check'],default='all');p.add_argument('--out',type=Path);a=p.parse_args()
 if a.mode in ['all','experiment']:
  if a.out is None:p.error('--out required; use a fresh directory')
  if a.out.exists():p.error('output already exists')
 if a.mode=='hand':result=hand()
 elif a.mode=='data':result=dict(cases=len(CASES),profiles=CASES,scope='scripted planner; real child-process death; SQLite checkpoints; real in-process MCP in workers')
 elif a.mode=='check':
  with tempfile.TemporaryDirectory() as d:result=check(Path(d)/'checks')
 else:
  result=run(a.out)
  if a.mode=='all':
   result['checks']=check(a.out/'checks',result);result['environment']={n:version(n) for n in ['mcp','mcp-types','pydantic','anyio','jsonschema','httpx','fastapi']}
   result['environment'].update(python=platform.python_version(),sqlite=sqlite3.sqlite_version,machine=platform.machine(),device='cpu')
   (a.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
