import argparse,json,tempfile,platform,sqlite3
from pathlib import Path
from importlib.metadata import version
from experiment53 import run,CASES
from checks53 import check

def hand():return dict(proposal_at=100,approval_expires_at=160,execute_at=161,approval_valid=False,target_effects_after_lost_ack=1,local_receipt_before_status=0,post_retries_same_key=2,note_effects_with_dedup=1,scope='synthetic fixture clock/counters; not real timeout latency or exactly-once end-to-end')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','experiment','check'],default='all');p.add_argument('--out',type=Path);a=p.parse_args()
 if a.mode in ['all','experiment']:
  if a.out is None:p.error('--out required; choose a fresh directory')
  if a.out.exists():p.error('output already exists')
 if a.mode=='hand':result=hand()
 elif a.mode=='data':result=dict(cases=len(CASES),profiles=CASES,scope='one real MCP investigation replayed into synthetic review/write cases; real SQLite effects via in-process HTTP-ASGI')
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
