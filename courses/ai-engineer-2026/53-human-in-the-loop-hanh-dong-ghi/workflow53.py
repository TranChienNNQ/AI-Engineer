import copy,hashlib,html,json,os,sqlite3
from pathlib import Path
from contextlib import contextmanager,ExitStack
from bridge53 import (Fault,digest,canonical,loads,World48,Link,MCPWorld,Journal,MemoryStore,ContextAgent,source_hash)
from contracts53 import Draft,Receipt,signed
from connector53 import Connector,UnknownOutcome
from target53 import Target
class ReviewError(ValueError):pass

def implementation_hash():
 root=Path(__file__).parent
 return digest({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['bridge53.py','contracts53.py','target53.py','connector53.py','workflow53.py','worker.py']})
def release_id():return digest(dict(base52=source_hash(),implementation53=implementation_hash()))

class SourceGate:
 def __init__(self,directory,revoke=False,now=100):self.directory=Path(directory);self.revoke=revoke;self.now=now
 def check(self):
  d=self.directory;cfg=loads((d/'config.json').read_text());j=Journal(d/'state.sqlite');m=MemoryStore(d/'memory.sqlite')
  try:
   with Link(d/'world.sqlite') as link:
    w=MCPWorld(d/'world.sqlite',link)
    try:
     # Teaching control plane shares a synthetic identity registry with target API.
     t=Target(d/'target.sqlite')
     try:
      for row in t.db.execute('SELECT * FROM actors'):w.core.identities[row['token']]=dict(id=row['id'],tenant=row['tenant'],role=row['role'],active=bool(row['active']))
     finally:t.close()
     if self.revoke:w.core.revoked.add('gateway')
     a=ContextAgent(j,w,'run-a',cfg['task'],'demo-agent-a',limits=cfg['limits'],expires_at=160,memory=m,policy=cfg['policy'],now=self.now)
     loaded=j.load('run-a','a:agent-a');s=loaded['state']
     if loaded['header']['binding']!=canonical(a.binding):raise ReviewError('source_binding_changed')
     a.current(s)
     if s['phase']!='answered':raise ReviewError('investigation_not_answered')
     quote=s['result']['citations'][0];health=next(o['data'] for o in s['observations'] if o['tool']=='read_health' and o['ok']);ticket=next(o['data'] for o in s['observations'] if o['tool']=='read_ticket' and o['ok'])
     body=f"Runbook [{quote['doc_id']}@v{quote['version']}]: {quote['quote']}\nHealth snapshot {health['observed_at']}: {health['status']}; chưa xác định nguyên nhân gốc. Chưa thực hiện khắc phục."
     return dict(source_binding=digest(a.binding),evidence_hash=digest(s['observations']),base_body=body,ticket_id=ticket['id'],expected_version=ticket['version'],citations=s['result']['citations'])
    finally:w.close()
  finally:j.close();m.close()

class Workflow:
 def __init__(self,directory,token='demo-agent-a',now=100,fault=None,revoke=False):
  self.directory=Path(directory);self.token=token;self.now=now;self.connector=Connector(self.directory/'target.sqlite',token,now,fault);self.gate=SourceGate(directory,revoke,now)
  self.db=sqlite3.connect(self.directory/'approval.sqlite',isolation_level=None,autocommit=sqlite3.LEGACY_TRANSACTION_CONTROL,timeout=2);self.db.row_factory=sqlite3.Row;self.db.execute('PRAGMA synchronous=FULL')
  self.db.executescript('''CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,tenant TEXT,actor TEXT,revision INTEGER,state TEXT,head_hash TEXT);
CREATE TABLE IF NOT EXISTS events(op_id TEXT,revision INTEGER,kind TEXT,snapshot TEXT,previous_hash TEXT,hash TEXT,PRIMARY KEY(op_id,revision));''')
 def close(self):self.db.close()
 def principal(self):
  t=Target(self.directory/'target.sqlite')
  try:return t.principal(self.token)
  finally:t.close()
 @contextmanager
 def tx(self):
  self.db.execute('BEGIN IMMEDIATE')
  try:yield;self.db.execute('COMMIT')
  except BaseException:
   if self.db.in_transaction:self.db.execute('ROLLBACK')
   raise
 def _load(self,op_id,p):
  head=self.db.execute('SELECT * FROM jobs WHERE id=?',(op_id,)).fetchone()
  if not head or head['tenant']!=p['tenant'] or p['id']!=head['actor'] and p['role']!='reviewer':raise ReviewError('operation_scope_denied')
  rows=self.db.execute('SELECT * FROM events WHERE op_id=? ORDER BY revision',(op_id,)).fetchall();prev=None
  if len(rows)!=head['revision'] or len(rows)>100:raise ReviewError('event_chain')
  for i,row in enumerate(rows,1):
   data=dict(op_id=op_id,revision=i,kind=row['kind'],snapshot=loads(row['snapshot']),previous_hash=prev)
   if row['revision']!=i or row['previous_hash']!=prev or digest(data)!=row['hash']:raise ReviewError('event_hash')
   prev=row['hash']
  s=loads(head['state'])
  if not rows or prev!=head['head_hash'] or s!=loads(rows[-1]['snapshot']):raise ReviewError('event_head')
  if digest(Draft.model_validate(s['draft']).model_dump())!=s['draft_hash']:raise ReviewError('draft_hash')
  return dict(state=s,revision=head['revision'],hash=prev)
 def load(self,op_id):
  p=self.principal();self.db.execute('BEGIN')
  try:return self._load(op_id,p)
  finally:self.db.execute('ROLLBACK')
 def _save(self,op_id,p,loaded,s,kind):
  fresh=self._load(op_id,p)
  if fresh['revision']!=loaded['revision']:raise ReviewError('stale_workflow_revision')
  old=fresh['state']
  if s['attempts']<old['attempts'] or s['status_checks']<old['status_checks']:raise ReviewError('counter_regression')
  if old['attempts'] and (s['draft']!=old['draft'] or s['draft_hash']!=old['draft_hash']):raise ReviewError('attempted_payload_frozen')
  if old['phase']=='confirmed' and s!=old:raise ReviewError('confirmed_immutable')
  return self._append(op_id,p,loaded['revision']+1,fresh['hash'],s,kind)
 def _append(self,op_id,p,revision,previous,s,kind):
  hash_=digest(dict(op_id=op_id,revision=revision,kind=kind,snapshot=s,previous_hash=previous))
  self.db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',(op_id,revision,kind,canonical(s),previous,hash_))
  self.db.execute('UPDATE jobs SET revision=?,state=?,head_hash=? WHERE id=?',(revision,canonical(s),hash_,op_id))
  return dict(state=copy.deepcopy(s),revision=revision,hash=hash_)
 def create(self,op_id='note-a',proposal=None):
  import re
  p=self.principal()
  if p['role']!='agent' or re.fullmatch(r'[a-zA-Z0-9_-]{1,60}',op_id) is None:raise ReviewError('agent_operation_required')
  src=self.gate.check();draft=Draft.model_validate(proposal or dict(version=1,operation='append_internal_note',ticket_id=src['ticket_id'],expected_version=src['expected_version'],service='gateway',body=src['base_body'],source_binding=src['source_binding'],evidence_hash=src['evidence_hash'],policy_release=release_id(),expires_at=200)).model_dump()
  self.validate_source(draft,src,src['base_body']);self.connector.ticket(draft['ticket_id'])
  s=dict(phase='awaiting_review',draft=draft,draft_hash=digest(draft),base_body=src['base_body'],citations=src['citations'],approval=None,attempts=0,status_checks=0,receipt=None,last_code='review_required')
  with self.tx():
   if self.db.execute('SELECT 1 FROM jobs WHERE id=?',(op_id,)).fetchone():raise ReviewError('operation_exists')
   self.db.execute('INSERT INTO jobs VALUES(?,?,?,0,?,NULL)',(op_id,p['tenant'],p['id'],canonical(s)));return self._append(op_id,p,1,None,s,'proposed')
 def validate_source(self,draft,src,base_body):
  if draft['policy_release']!=release_id():raise ReviewError('policy_release_changed')
  if (draft['ticket_id'],draft['expected_version'],draft['source_binding'],draft['evidence_hash'])!=(src['ticket_id'],src['expected_version'],src['source_binding'],src['evidence_hash']):raise ReviewError('draft_source_scope')
  if base_body!=src['base_body'] or draft['body']!=base_body and not draft['body'].startswith(base_body+'\nReviewer comment: '):raise ReviewError('unsupported_draft_body')
 def check_current(self,s):
  src=self.gate.check();self.validate_source(s['draft'],src,s['base_body'])
  if self.now>=s['draft']['expires_at']:raise ReviewError('proposal_expired')
 def view(self,op_id='note-a'):
  row=self.load(op_id);s=row['state'];self.check_current(s);before=self.connector.ticket(s['draft']['ticket_id'])
  view=dict(op_id=op_id,workflow_revision=row['revision'],draft_hash=s['draft_hash'],phase=s['phase'],draft=s['draft'],current_target=before,approval_expires_at_if_approved=min(self.now+60,s['draft']['expires_at']),citations=s['citations'],impact='Append one internal note; keep ticket open. No remediation or customer notification.',uncertainty='Root cause and remediation have not been verified.')
  return view|dict(display_hash=digest(view))
 def review(self,op_id,expected_revision,display_hash,decision='approve'):
  p=self.principal();view=self.view(op_id);loaded=self.load(op_id);s=copy.deepcopy(loaded['state'])
  if p['role']!='reviewer':raise ReviewError('reviewer_required')
  head=self.db.execute('SELECT actor FROM jobs WHERE id=?',(op_id,)).fetchone()
  if head['actor']==p['id']:raise ReviewError('self_review_denied')
  if decision not in ['approve','reject']:raise ReviewError('invalid_decision')
  if loaded['revision']!=expected_revision or view['display_hash']!=display_hash:raise ReviewError('stale_review_view')
  if s['phase']!='awaiting_review':raise ReviewError('not_awaiting_review')
  if view['current_target']['version']!=s['draft']['expected_version'] or view['current_target']['status']!='open':raise ReviewError('target_changed_before_review')
  if decision=='reject':s.update(phase='rejected',last_code='review_rejected')
  else:s.update(phase='approved',approval=dict(reviewer=p['id'],tenant=p['tenant'],draft_hash=s['draft_hash'],draft_version=s['draft']['version'],expires_at=min(self.now+60,s['draft']['expires_at'])),last_code='approved_not_executed')
  with self.tx():return self._save(op_id,p,loaded,s,'review_'+decision)
 def edit(self,op_id,expected_revision,comment):
  p=self.principal();loaded=self.load(op_id);s=copy.deepcopy(loaded['state'])
  if p['role']!='reviewer':raise ReviewError('reviewer_required')
  if loaded['revision']!=expected_revision:raise ReviewError('stale_workflow_revision')
  if s['attempts'] or s['phase'] not in ['awaiting_review','approved']:raise ReviewError('attempted_payload_frozen')
  self.check_current(s)
  if not isinstance(comment,str) or not 1<=len(comment)<=160:raise ReviewError('comment_size')
  s['draft']=Draft.model_validate(s['draft']|dict(version=s['draft']['version']+1,body=s['base_body']+'\nReviewer comment: '+comment)).model_dump();s.update(draft_hash=digest(s['draft']),approval=None,phase='awaiting_review',last_code='edited_requires_new_review')
  with self.tx():return self._save(op_id,p,loaded,s,'edited')
 def cancel(self,op_id):
  p=self.principal();loaded=self.load(op_id);s=copy.deepcopy(loaded['state'])
  if s['attempts'] or s['phase'] not in ['awaiting_review','approved']:raise ReviewError('reconcile_before_cancel')
  s.update(phase='cancelled',approval=None,last_code='cancelled_before_send')
  with self.tx():return self._save(op_id,p,loaded,s,'cancelled')
 def permit(self,op_id,s):
  p=self.principal();approval=s['approval']
  if p['role']!='agent':raise ReviewError('agent_required')
  if not approval or approval['draft_hash']!=s['draft_hash'] or approval['draft_version']!=s['draft']['version']:raise ReviewError('approval_required')
  if self.now>=approval['expires_at']:raise ReviewError('approval_expired')
  t=Target(self.directory/'target.sqlite')
  try:reviewer=t.by_id(approval['reviewer'])
  finally:t.close()
  if reviewer['role']!='reviewer' or reviewer['tenant']!=p['tenant']:raise ReviewError('reviewer_unavailable')
  return signed(dict(op_id=op_id,draft_hash=s['draft_hash'],tenant=p['tenant'],actor=p['id'],reviewer=reviewer['id'],expires_at=approval['expires_at']))
 def receipt(self,op_id,s,data):
  receipt=Receipt.model_validate(data).model_dump()
  if (receipt['op_id'],receipt['draft_hash'],receipt['ticket_id'],receipt['version'])!=(op_id,s['draft_hash'],s['draft']['ticket_id'],s['draft']['expected_version']+1):raise ReviewError('receipt_identity')
  return receipt
 def execute(self,op_id='note-a',retry_same_key=False,crash_at=None):
  p=self.principal();loaded=self.load(op_id);s=copy.deepcopy(loaded['state'])
  if s['phase']=='confirmed':return loaded
  if s['phase']!='approved' and not (s['phase']=='unknown' and retry_same_key):raise ReviewError('approval_or_reconciliation_required')
  if s['phase']=='unknown' and s['status_checks']==0:raise ReviewError('reconcile_required_before_retry')
  self.check_current(s);permit=self.permit(op_id,s)
  if s['attempts']>=2:raise ReviewError('write_attempt_budget')
  s['attempts']+=1;s.update(phase='in_flight',last_code='write_reserved')
  with self.tx():loaded=self._save(op_id,p,loaded,s,'write_reserved')
  if crash_at=='before_send':os._exit(83)
  try:
   data=self.connector.apply(dict(op_id=op_id,draft=s['draft'],**permit))
   if crash_at=='after_response':os._exit(83)
   s.update(receipt=self.receipt(op_id,s,data),phase='confirmed',last_code='write_confirmed')
  except (UnknownOutcome,ValueError,ReviewError):s.update(phase='unknown',receipt=None,last_code='write_outcome_unknown')
  except Fault as e:s.update(phase='failed',last_code=e.code)
  with self.tx():return self._save(op_id,p,loaded,s,'write_result')
 def reconcile(self,op_id='note-a'):
  p=self.principal();loaded=self.load(op_id);s=copy.deepcopy(loaded['state'])
  if s['phase']=='confirmed':return loaded
  if p['role']!='agent' or s['phase'] not in ['unknown','in_flight']:raise ReviewError('nothing_to_reconcile')
  if s['status_checks']>=3:raise ReviewError('status_check_budget')
  s['status_checks']+=1;s.update(phase='unknown',last_code='status_reserved')
  with self.tx():loaded=self._save(op_id,p,loaded,s,'status_reserved')
  # Approval expiry/source change cannot erase an effect that may already have happened.
  try:
   status=self.connector.status(op_id)
   if status['state']=='applied':s.update(receipt=self.receipt(op_id,s,status['receipt']),phase='confirmed',last_code='confirmed_by_status')
   else:s['last_code']='status_not_found_still_unknown'
  except (UnknownOutcome,ValueError,ReviewError):s['last_code']='status_unavailable_still_unknown'
  with self.tx():return self._save(op_id,p,loaded,s,'status_result')

def render_review(view,path):
 # Read-only artifact; no executable approval buttons or model-generated HTML.
 escaped=html.escape(json.dumps(view,ensure_ascii=False,indent=2));Path(path).write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Duyệt ghi chú nội bộ</title><body><h1>Duyệt hành động ghi</h1><p>Đọc đúng payload, target/version, nguồn và expiry. Approve là cấp quyền một hành động, chưa là đã thực thi.</p><pre>'+escaped+'</pre></body></html>')
