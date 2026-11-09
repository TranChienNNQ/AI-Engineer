import argparse,json,tempfile,platform,sqlite3
from pathlib import Path
from importlib.metadata import version
from experiment52 import run,CASES
from checks52 import check

def hand():return dict(window_bytes=14000,reserve_bytes=2000,available_input_bytes=12000,compact_warm_input_bytes=25079,full_warm_input_bytes=26206,difference_bytes=1127,cold_tool_attempts=4,warm_tool_attempts=3,scope='serialized JSON UTF-8 byte counts, not model tokens/cost; scripted policy')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','experiment','check'],default='all');p.add_argument('--out',type=Path);a=p.parse_args()
 if a.mode in ['all','experiment']:
  if a.out is None:p.error('--out required; use a fresh directory')
  if a.out.exists():p.error('output already exists')
 if a.mode=='hand':result=hand()
 elif a.mode=='data':result=dict(cases=len(CASES),profiles=CASES,cross_run_demo=True,scope='scripted planner/summary; real scoped SQLite memory, local durable journal, real MCP reads')
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
