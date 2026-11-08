import json,sqlite3,subprocess,sys,hashlib
from pathlib import Path
from bridge51 import b,digest,Link,MCPWorld,task,source_hash
from journal51 import Journal
from runtime51 import DurableAgent,implementation_hash
from contracts51 import Limits
ROOT=Path(__file__).resolve().parent

def setup(directory,run_id='support-504',ambiguous=True,limits=None):
 d=Path(directory);d.mkdir(parents=True,exist_ok=False)
 # Trusted fixture preparation, before agent execution; no write tool is exposed.
 w=b.World48(d/'world.sqlite')
 body=json.dumps(dict(symptom='gateway timeout 504 upstream trace_id',reported_health='degraded',report_at='2026-11-08T02:00:00Z'),ensure_ascii=False,sort_keys=True)
 try:
  w.core.store.db.execute('UPDATE tickets SET body=? WHERE id=1',(body,));seed=digest([dict(r) for r in w.core.store.db.execute('SELECT * FROM tickets ORDER BY id')])
 finally:w.close()
 t=task(question='mã XQ7 ở cổng khách hàng' if ambiguous else 'gateway timeout 504 upstream trace_id')
 config=dict(run_id=run_id,task=t,limits=Limits.model_validate(limits or {}).model_dump(),expires_at=160,seed_hash=seed)
 (d/'config.json').write_text(json.dumps(config,ensure_ascii=False,sort_keys=True)+'\n')
 with Link(d/'world.sqlite') as link:
  world=MCPWorld(d/'world.sqlite',link);j=Journal(d/'state.sqlite')
  try:DurableAgent(j,world,run_id,t,'demo-agent-a',limits=config['limits'],expires_at=160).create()
  finally:j.close();world.close()
 return config

def launch(directory,owner='worker1',**options):
 cmd=[sys.executable,str(ROOT/'worker.py'),'--directory',str(directory),'--owner',owner]
 for key,value in options.items():
  if value is not None:cmd.extend(['--'+key.replace('_','-'),str(value)])
 done=subprocess.run(cmd,capture_output=True,text=True,timeout=25,cwd=ROOT)
 if done.returncode not in [0,83,84]:raise AssertionError(done.stderr)
 return dict(exit=done.returncode,output=json.loads(done.stdout) if done.returncode==0 else None)

def inspect(directory):
 d=Path(directory);config=json.loads((d/'config.json').read_text());j=Journal(d/'state.sqlite')
 try:loaded=j.load(config['run_id'],'a:agent-a')
 finally:j.close()
 db=sqlite3.connect(d/'probe.sqlite') if (d/'probe.sqlite').exists() else None
 acks=[] if db is None else [dict(zip(['n','run_id','op_id','attempt','epoch','result_hash'],r)) for r in db.execute('SELECT * FROM acknowledgements ORDER BY n')]
 if db:db.close()
 world=sqlite3.connect(d/'world.sqlite');world.row_factory=sqlite3.Row
 seed=digest([dict(r) for r in world.execute('SELECT * FROM tickets ORDER BY id')]);effects={n:world.execute('SELECT COUNT(*) FROM '+n).fetchone()[0] for n in ['tickets','runs','proposals','receipts','audit']};world.close()
 return dict(state=loaded['state'],header={k:v for k,v in loaded['header'].items() if k!='binding'},checkpoints=[dict(revision=r['revision'],kind=r['kind'],epoch=r['epoch'],hash=r['hash']) for r in loaded['history']],acknowledgements=acks,seed_unchanged=seed==config['seed_hash'],effects=effects)

CASES=[dict(id='uninterrupted'),dict(id='pause_resume',pause_after=2),
 *[dict(id=point,crash_at=point) for point in ['planning_reserved','intent_committed','attempt_reserved','after_response','observation_committed','terminal_transaction','terminal_committed']],
 dict(id='attempt_budget',crash_at='after_response',limits=dict(attempts=2)),
 dict(id='per_op_budget',crash_at='after_response',limits=dict(per_operation=1)),
 dict(id='expired',pause_after=2,resume_now=161),dict(id='turn_budget',crash_at='planning_reserved',limits=dict(turns=1))]

def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);rows=[]
 for c in CASES:
  d=out/c['id'];config=setup(d,limits=c.get('limits'));first=launch(d,**{k:c[k] for k in ['crash_at','pause_after'] if k in c});before=inspect(d)
  second=None
  if 'crash_at' in c or 'pause_after' in c:second=launch(d,owner='worker2',takeover_epoch=before['header']['epoch'] if before['header']['owner'] else None,now=c.get('resume_now',100))
  after=inspect(d);rows.append(dict(id=c['id'],first=first,second=second,before=before,after=after))
 manifest=dict(base50=source_hash(),implementation51=implementation_hash(),experiment_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=digest(CASES))
 result=dict(cases=len(rows),rows=rows,manifest=manifest,experiment_id=digest(manifest)[:16],provider_http_attempts=0,live_llm_calls=0,
             completed_acknowledgements=sum(len(r['after']['acknowledgements']) for r in rows),attempt_reservations=sum(r['after']['state']['attempts'] for r in rows),planning_reservations=sum(r['after']['state']['turns'] for r in rows),process_launches=sum(1+(r['second'] is not None) for r in rows))
 (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result
