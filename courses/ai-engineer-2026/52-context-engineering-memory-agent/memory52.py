"""Scoped, reviewed memory with source lineage; not an autonomous memory writer."""
import sqlite3,copy,re
from contextlib import contextmanager
from bridge52 import digest,loads,canonical
from contracts52 import Proposal,PAYLOADS,namespace
class MemoryError(ValueError):pass
class MemoryStore:
 def __init__(self,path):
  self.db=sqlite3.connect(path,timeout=2,isolation_level=None,autocommit=sqlite3.LEGACY_TRANSACTION_CONTROL);self.db.row_factory=sqlite3.Row
  self.db.execute('PRAGMA synchronous=FULL')
  self.db.executescript('''CREATE TABLE IF NOT EXISTS meta(id INTEGER PRIMARY KEY CHECK(id=1),epoch INTEGER NOT NULL);
INSERT OR IGNORE INTO meta VALUES(1,0);
CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY,tenant TEXT,owner TEXT,service TEXT,origin_run TEXT,kind TEXT,version INTEGER,payload TEXT,hash TEXT,revoked INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY,tenant TEXT,owner TEXT,service TEXT,key TEXT,kind TEXT,payload TEXT,refs TEXT,created_at INTEGER,expires_at INTEGER,status TEXT);
CREATE TABLE IF NOT EXISTS tombstones(tenant TEXT,owner TEXT,key TEXT,PRIMARY KEY(tenant,owner,key));
CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY,operation TEXT,memory_id INTEGER,reason TEXT);''')
 def close(self):self.db.close()
 @property
 def epoch(self):return self.db.execute('SELECT epoch FROM meta WHERE id=1').fetchone()[0]
 @contextmanager
 def tx(self):
  self.db.execute('BEGIN IMMEDIATE')
  try:yield;self.db.execute('UPDATE meta SET epoch=epoch+1 WHERE id=1');self.db.execute('COMMIT')
  except BaseException:
   if self.db.in_transaction:self.db.execute('ROLLBACK')
   raise
 def reviewer(self,p,tenant):
  if not p.get('active') or p.get('role')!='reviewer' or p['tenant']!=tenant:raise MemoryError('review_scope_denied')
 def register_source(self,reviewer,owner,source_id,service,origin_run,kind,payload):
  # Trusted source registration, NOT a tool exposed to the planner.
  self.reviewer(reviewer,owner['tenant'])
  if kind not in ['alias','report','preference','episode']:raise MemoryError('source_kind')
  with self.tx():
   old=self.db.execute('SELECT * FROM sources WHERE id=?',(source_id,)).fetchone()
   if old and (old['tenant'],old['owner'])!=(owner['tenant'],owner['id']):raise MemoryError('source_namespace')
   version=old['version']+1 if old else 1
   self.db.execute('INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?,?,?,?,0)',(source_id,owner['tenant'],owner['id'],service,origin_run,kind,version,canonical(payload),digest(payload)))
  return dict(source_id=source_id,version=version,content_hash=digest(payload))
 def source_rows(self,p,service,refs):
  rows=[]
  for ref in refs:
   row=self.db.execute('SELECT * FROM sources WHERE id=?',(ref['source_id'],)).fetchone()
   if not row or (row['tenant'],row['owner'],row['service'])!=(p['tenant'],p['id'],service):raise MemoryError('source_scope')
   if row['revoked']:raise MemoryError('source_revoked')
   if row['version']!=ref['version'] or row['hash']!=ref['content_hash'] or digest(loads(row['payload']))!=row['hash']:raise MemoryError('source_stale')
   rows.append(dict(row))
  if len({r['id'] for r in rows})!=len(rows) or len({r['origin_run'] for r in rows})!=1:raise MemoryError('source_lineage')
  return rows
 def validate(self,p,service,proposal,now):
  try:
   q=Proposal.model_validate(proposal).model_dump();q['payload']=PAYLOADS[q['kind']].model_validate(q['payload']).model_dump()
  except ValueError:raise MemoryError('proposal_schema') from None
  if q['expires_at']<=now:raise MemoryError('expired_proposal')
  if self.db.execute('SELECT 1 FROM tombstones WHERE tenant=? AND owner=? AND key=?',(p['tenant'],p['id'],q['key'])).fetchone():raise MemoryError('forgotten_key')
  rows=self.source_rows(p,service,q['refs'])
  required={'alias','report'} if q['kind']=='hint' else {q['kind']}
  if {r['kind'] for r in rows}!=required or len(rows)!=len(required):raise MemoryError('source_lineage')
  bykind={r['kind']:loads(r['payload']) for r in rows}
  if q['kind']=='hint':expected=dict(trigger=bykind.get('alias',{}).get('trigger'),query=bykind.get('report',{}).get('symptom'))
  else:expected=bykind.get(q['kind'])
  if q['payload']!=expected:raise MemoryError('payload_not_supported')
  return q
 def stage(self,p,service,proposal,now=100):
  if not p.get('active'):raise MemoryError('inactive_actor')
  with self.tx():
   q=self.validate(p,service,proposal,now)
   cur=self.db.execute('INSERT INTO memories(tenant,owner,service,key,kind,payload,refs,created_at,expires_at,status) VALUES(?,?,?,?,?,?,?,?,?,?)',(p['tenant'],p['id'],service,q['key'],q['kind'],canonical(q['payload']),canonical(q['refs']),now,q['expires_at'],'candidate'));id_=cur.lastrowid
   self.db.execute('INSERT INTO decisions(operation,memory_id,reason) VALUES(?,?,?)',('stage',id_,'source_checked'))
  return id_
 def approve(self,reviewer,id_,now=100):
  with self.tx():
   row=self.db.execute('SELECT * FROM memories WHERE id=?',(id_,)).fetchone()
   if not row:raise MemoryError('memory_not_found')
   self.reviewer(reviewer,row['tenant'])
   if row['status']!='candidate':raise MemoryError('not_candidate')
   self.validate(dict(tenant=row['tenant'],id=row['owner']),row['service'],self.proposal(row),now)
   self.db.execute("UPDATE memories SET status='active' WHERE id=?",(id_,));self.db.execute('INSERT INTO decisions(operation,memory_id,reason) VALUES(?,?,?)',('approve',id_,'trusted_fixture_review'))
 def proposal(self,row):return {k:(loads(row[k]) if k in ['payload','refs'] else row[k]) for k in ['key','kind','payload','refs','expires_at']}
 def _recall(self,p,service,question,now=100):
  if not p.get('active'):raise MemoryError('inactive_actor')
  # Scope BEFORE ranking. Foreign namespaces never enter the candidate pool or diagnostics.
  rows=self.db.execute('SELECT * FROM memories WHERE tenant=? AND owner=? AND service=? ORDER BY id',(p['tenant'],p['id'],service)).fetchall();eligible=[];excluded=[]
  for row in rows:
   reason=None
   if row['status']!='active':reason=row['status']
   elif row['expires_at']<=now:reason='expired'
   else:
    try:q=self.validate(p,service,self.proposal(row),now)
    except MemoryError as e:reason=str(e)
   if reason:excluded.append(dict(id=row['id'],reason=reason));continue
   if q['kind']=='hint' and re.search(r'(?<!\w)'+re.escape(q['payload']['trigger'])+r'(?!\w)',question,re.I) is None:
    excluded.append(dict(id=row['id'],reason='irrelevant'));continue
   eligible.append(dict(id=row['id'],key=q['key'],kind=q['kind'],payload=q['payload'],refs=q['refs'],created_at=row['created_at'],expires_at=row['expires_at'],score={'hint':100,'preference':50,'episode':20}[q['kind']]))
  conflicted={x['key'] for x in eligible if len({digest(v['payload']) for v in eligible if v['key']==x['key']})>1}
  selected=[];seen=set()
  for x in sorted(eligible,key=lambda m:(-m['score'],-m['created_at'],m['id'])):
   if x['key'] in conflicted:excluded.append(dict(id=x['id'],reason='conflict'))
   elif x['key'] in seen:excluded.append(dict(id=x['id'],reason='duplicate'))
   else:selected.append(x);seen.add(x['key'])
  return dict(epoch=self.epoch,items=selected,excluded=excluded)
 def recall(self,p,service,question,now=100):
  self.db.execute('BEGIN')
  try:return self._recall(p,service,question,now)
  finally:self.db.execute('ROLLBACK')
 def resolve(self,reviewer,winner,expected_epoch):
  with self.tx():
   if self.epoch!=expected_epoch:raise MemoryError('stale_memory_epoch')
   row=self.db.execute('SELECT * FROM memories WHERE id=?',(winner,)).fetchone()
   if not row:raise MemoryError('memory_not_found')
   self.reviewer(reviewer,row['tenant'])
   if row['status']!='active':raise MemoryError('winner_not_active')
   self.db.execute("UPDATE memories SET status='superseded' WHERE tenant=? AND owner=? AND key=? AND id!=? AND status='active'",(row['tenant'],row['owner'],row['key'],winner))
 def revoke_source(self,reviewer,source_id):
  with self.tx():
   row=self.db.execute('SELECT * FROM sources WHERE id=?',(source_id,)).fetchone()
   if not row:raise MemoryError('source_not_found')
   self.reviewer(reviewer,row['tenant']);self.db.execute('UPDATE sources SET revoked=1 WHERE id=?',(source_id,))
 def forget(self,p,key):
  if not p.get('active'):raise MemoryError('inactive_actor')
  with self.tx():
   self.db.execute('INSERT OR IGNORE INTO tombstones VALUES(?,?,?)',(p['tenant'],p['id'],key))
   self.db.execute("UPDATE memories SET status='forgotten',payload='{}',refs='[]' WHERE tenant=? AND owner=? AND key=?",(p['tenant'],p['id'],key))
