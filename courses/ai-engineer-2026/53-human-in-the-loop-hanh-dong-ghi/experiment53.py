import json,sqlite3,subprocess,sys,hashlib,shutil
from pathlib import Path
from bridge53 import World48,MemoryStore,b,canonical,digest,loads,source_hash
from target53 import Target
from workflow53 import Workflow,release_id,implementation_hash
ROOT=Path(__file__).resolve().parent

def template(path,warm=False):
 d=Path(path);d.mkdir(parents=True,exist_ok=False);w=World48(d/'world.sqlite')
 try:
  body=json.dumps(dict(symptom='gateway timeout 504 upstream trace_id',reported_health='degraded',report_at='2026-11-08T02:00:00Z'),ensure_ascii=False,sort_keys=True)
  owner=w.principal('demo-agent-a');reviewer=w.principal('demo-reviewer-a')
  w.core.store.db.execute('UPDATE tickets SET body=? WHERE id=1',(body,));seed=digest([dict(r) for r in w.core.store.db.execute('SELECT * FROM tickets ORDER BY id')])
 finally:w.close()
 policy=sys.modules['contracts52'].ContextPolicy().model_dump();cfg=dict(task=b.task(question='mã XQ7 ở cổng khách hàng'),limits=b.Limits().model_dump(),policy=policy,seed_hash=seed)
 (d/'config.json').write_text(canonical(cfg))
 m=MemoryStore(d/'memory.sqlite')
 if warm:
  payload=dict(language='vi');ref=m.register_source(reviewer,owner,'user-language','gateway','user-confirmation','preference',payload)
  id_=m.stage(owner,'gateway',dict(key='language',kind='preference',payload=payload,refs=[ref],expires_at=150));m.approve(reviewer,id_)
 m.close()
 done=subprocess.run([sys.executable,str(ROOT.parent/'52-context-engineering-memory-agent/worker.py'),'--directory',str(d),'--create'],capture_output=True,text=True,timeout=30)
 if done.returncode or not json.loads(done.stdout)['ok']:raise AssertionError(done.stderr or done.stdout)
 return json.loads(done.stdout)

def setup(directory,seed):
 d=Path(directory);d.mkdir(parents=True,exist_ok=False);seed=Path(seed)
 for name in ['world','state','memory','probe']:
  a=sqlite3.connect(seed/(name+'.sqlite'));z=sqlite3.connect(d/(name+'.sqlite'))
  try:a.backup(z)
  finally:a.close();z.close()
 shutil.copyfile(seed/'config.json',d/'config.json');t=Target(d/'target.sqlite');t.close();return d

def launch(directory,action,**options):
 cmd=[sys.executable,str(ROOT/'worker.py'),'--directory',str(directory),'--action',action]
 for k,v in options.items():
  if v is True:cmd.append('--'+k.replace('_','-'))
  elif v is not None and v is not False:cmd.extend(['--'+k.replace('_','-'),str(v)])
 done=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=30)
 if done.returncode not in [0,83]:raise AssertionError(done.stderr)
 return dict(exit=done.returncode,output=json.loads(done.stdout) if done.returncode==0 else None)

def reviewed(directory,decision='approve',**kwargs):
 view=launch(directory,'view',token='demo-reviewer-a')['output']['result']
 result=launch(directory,decision,token='demo-reviewer-a',expected_revision=view['workflow_revision'],display_hash=view['display_hash'],**kwargs)
 return view,result

def inspect(directory):
 d=Path(directory);w=Workflow(d)
 try:row=w._load('note-a',dict(tenant='a',id='agent-a',role='agent'));events=[dict(v) for v in w.db.execute('SELECT * FROM events ORDER BY revision')]
 finally:w.close()
 t=Target(d/'target.sqlite')
 try:
  target=dict(tickets=[dict(v) for v in t.db.execute('SELECT * FROM tickets ORDER BY id')],notes=[dict(v) for v in t.db.execute('SELECT * FROM notes ORDER BY id')],receipts=[dict(v) for v in t.db.execute('SELECT * FROM receipts ORDER BY op_id')],requests=[dict(v) for v in t.db.execute('SELECT * FROM requests ORDER BY id')])
 finally:t.close()
 return dict(local=row,events=events,target=target,review_html=(d/'review.html').exists())

CASES=['approved','reject','cancel','no_approval','wrong_reviewer','stale_view','edited_requires_review','edited_reapproved','expired','stale_target','source_revoked','memory_changed','reviewer_revoked','actor_revoked','lost_ack','status_unavailable','status_missing_retry','before_send_retry','before_commit_retry','bad_receipt','crash_after_response','crash_before_send','late_reconcile','reconcile_after_source_revoke','cancel_unknown','attempt_budget']

def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);baseline=template(out/'investigation');rows=[]
 for name in CASES:
  d=setup(out/name,out/'investigation');steps=[launch(d,'prepare')];before=None
  if name in ['reject','cancel']:
   if name=='reject':view,result=reviewed(d,'reject');steps.append(result)
   else:steps.append(launch(d,'cancel'))
  elif name=='no_approval':steps.append(launch(d,'execute'))
  elif name=='wrong_reviewer':
   view=launch(d,'view',token='demo-reviewer-a')['output']['result'];steps.append(launch(d,'approve',expected_revision=view['workflow_revision'],display_hash=view['display_hash']))
  elif name=='stale_view':
   view=launch(d,'view',token='demo-reviewer-a')['output']['result'];steps.append(launch(d,'edit',token='demo-reviewer-a',expected_revision=view['workflow_revision'],comment='Kiểm tra upstream trước khi xử lý tiếp.'));steps.append(launch(d,'approve',token='demo-reviewer-a',expected_revision=view['workflow_revision'],display_hash=view['display_hash']))
  else:
   view,result=reviewed(d);steps.append(result)
   if name in ['edited_requires_review','edited_reapproved']:
    steps.append(launch(d,'edit',token='demo-reviewer-a',expected_revision=result['output']['result']['revision'],comment='Đề nghị bổ sung trace_id cho người trực.'))
    if name=='edited_reapproved':steps.append(reviewed(d)[1])
   if name=='stale_target':
    t=Target(d/'target.sqlite');t.db.execute('UPDATE tickets SET version=version+1 WHERE id=1');t.close()
   if name=='memory_changed':m=MemoryStore(d/'memory.sqlite');m.forget(dict(id='agent-a',tenant='a',active=True),'hint');m.close()
   if name=='actor_revoked':t=Target(d/'target.sqlite');t.db.execute("UPDATE actors SET active=0 WHERE id='agent-a'");t.close()
   if name=='reviewer_revoked':t=Target(d/'target.sqlite');t.db.execute("UPDATE actors SET active=0 WHERE id='reviewer-a'");t.close()
   kwargs=dict(now=161) if name=='expired' else dict(revoke=True) if name=='source_revoked' else {}
   fault={'lost_ack':'after_commit','status_unavailable':'after_commit','status_missing_retry':'after_commit','late_reconcile':'after_commit','reconcile_after_source_revoke':'after_commit','cancel_unknown':'after_commit','bad_receipt':'bad_receipt','before_send_retry':'before_send','before_commit_retry':'before_commit','attempt_budget':'before_send'}.get(name)
   if fault:kwargs['fault']=fault
   if name in ['crash_after_response','crash_before_send']:kwargs['crash_at']='after_response' if name=='crash_after_response' else 'before_send'
   steps.append(launch(d,'execute',**kwargs));before=inspect(d)
   if name in ['lost_ack','bad_receipt','crash_after_response','late_reconcile','reconcile_after_source_revoke']:
    steps.append(launch(d,'reconcile',now=161 if name=='late_reconcile' else 100,revoke=name=='reconcile_after_source_revoke'))
   if name=='status_unavailable':steps.append(launch(d,'reconcile',fault='status_unavailable'))
   if name in ['status_missing_retry','before_send_retry','before_commit_retry','crash_before_send']:
    steps.append(launch(d,'reconcile',fault='status_missing' if name=='status_missing_retry' else None));steps.append(launch(d,'execute',retry_same_key=True))
   if name=='cancel_unknown':steps.append(launch(d,'cancel'))
   if name=='approved':steps.append(launch(d,'execute'))
   if name=='attempt_budget':
    steps.append(launch(d,'reconcile'))
    steps.append(launch(d,'execute',retry_same_key=True,fault='before_send'));steps.append(launch(d,'execute',retry_same_key=True))
  rows.append(dict(id=name,steps=steps,before=before,after=inspect(d)))
 manifest=dict(base52=source_hash(),implementation53=implementation_hash(),release=release_id(),experiment_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=digest(CASES))
 result=dict(cases=len(rows),rows=rows,investigation=dict(state=baseline['result']['state'],attempt_reservations=baseline['result']['attempt_reservations'],turns=baseline['result']['turns']),manifest=manifest,experiment_id=digest(manifest)[:16],provider_http_attempts=0,live_llm_calls=0)
 (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result
