"""Single-file atomic snapshots + a hash chain, optimistic revision and owner epoch."""
import copy,sqlite3,os
from contextlib import contextmanager
from bridge51 import digest,loads
from contracts51 import State,canonical
class JournalError(ValueError):pass
class Journal:
 def __init__(self,path):
  self.db=sqlite3.connect(path,timeout=2,isolation_level=None,autocommit=sqlite3.LEGACY_TRANSACTION_CONTROL);self.db.row_factory=sqlite3.Row
  self.db.execute('PRAGMA foreign_keys=ON');self.db.execute('PRAGMA journal_mode=DELETE');self.db.execute('PRAGMA synchronous=FULL')
  self.db.executescript('''CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,actor TEXT NOT NULL,binding TEXT NOT NULL,head INTEGER NOT NULL,head_hash TEXT,owner TEXT,epoch INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS checkpoints(run_id TEXT NOT NULL REFERENCES runs(id),revision INTEGER NOT NULL,kind TEXT NOT NULL,epoch INTEGER NOT NULL,state TEXT NOT NULL,previous_hash TEXT,hash TEXT NOT NULL,PRIMARY KEY(run_id,revision));''')
 def close(self):self.db.close()
 @contextmanager
 def tx(self):
  self.db.execute('BEGIN IMMEDIATE')
  try:yield;self.db.execute('COMMIT')
  except BaseException:
   if self.db.in_transaction:self.db.execute('ROLLBACK')
   raise
 def _header(self,run_id,actor):
  row=self.db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
  if row is None:raise JournalError('run_not_found')
  if row['actor']!=actor:raise JournalError('run_actor_mismatch')
  return dict(row)
 def _load(self,run_id,actor):
  head=self._header(run_id,actor);rows=self.db.execute('SELECT * FROM checkpoints WHERE run_id=? ORDER BY revision',(run_id,)).fetchall();previous=None
  if not rows or len(rows)>128 or len(rows)!=head['head']:raise JournalError('checkpoint_chain')
  for i,row in enumerate(rows,1):
   data=dict(run_id=run_id,revision=i,kind=row['kind'],epoch=row['epoch'],state=loads(row['state']),previous_hash=previous)
   if row['revision']!=i or row['previous_hash']!=previous or digest(data)!=row['hash']:raise JournalError('checkpoint_hash')
   try:State.model_validate(data['state'])
   except ValueError:raise JournalError('checkpoint_schema') from None
   previous=row['hash']
  if previous!=head['head_hash']:raise JournalError('checkpoint_head')
  return dict(header=head,state=loads(rows[-1]['state']),history=[dict(r) for r in rows])
 def load(self,run_id,actor):
  # One consistent read transaction: do not mix old header and new snapshots.
  self.db.execute('BEGIN')
  try:return self._load(run_id,actor)
  finally:self.db.execute('ROLLBACK')
 def _append(self,header,state,kind,epoch,owner,crash=False):
  state=State.model_validate(state).model_dump();rev=header['head']+1
  if rev>128:raise JournalError('checkpoint_budget')
  data=dict(run_id=header['id'],revision=rev,kind=kind,epoch=epoch,state=state,previous_hash=header['head_hash']);hash_=digest(data)
  self.db.execute('INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?)',(header['id'],rev,kind,epoch,canonical(state),header['head_hash'],hash_))
  if crash:os._exit(84) # Test-only abrupt process death inside the open transaction.
  self.db.execute('UPDATE runs SET head=?,head_hash=?,epoch=?,owner=? WHERE id=?',(rev,hash_,epoch,owner,header['id']))
  return dict(revision=rev,epoch=epoch,owner=owner)
 def create(self,run_id,actor,binding,state=None):
  with self.tx():
   if self.db.execute('SELECT 1 FROM runs WHERE id=?',(run_id,)).fetchone():raise JournalError('run_exists')
   self.db.execute('INSERT INTO runs VALUES(?,?,?,0,NULL,NULL,0)',(run_id,actor,canonical(binding)))
   return self._append(self._header(run_id,actor),state or State().model_dump(),'created',0,None)
 def claim(self,run_id,actor,binding,owner,takeover_epoch=None):
  if not owner:raise JournalError('owner_required')
  with self.tx():
   loaded=self._load(run_id,actor);head=loaded['header']
   if loads(head['binding'])!=binding:raise JournalError('run_binding_changed')
   if head['owner'] is not None and takeover_epoch!=head['epoch']:raise JournalError('run_busy')
   # Recovery controller must first establish the prior process is dead. No lease detector here.
   return self._append(head,loaded['state'],'claimed',head['epoch']+1,owner)
 def save(self,run_id,actor,handle,state,kind,release=False,crash=False):
  with self.tx():
   loaded=self._load(run_id,actor);head=loaded['header'];old=loaded['state']
   if state['turns']<old['turns'] or state['attempts']<old['attempts']:raise JournalError('counter_regression')
   if state['observations'][:len(old['observations'])]!=old['observations'] or state['plans'][:len(old['plans'])]!=old['plans']:raise JournalError('history_rewrite')
   if old['result'] is not None and state!=old:raise JournalError('terminal_immutable')
   if head['owner']!=handle['owner'] or head['epoch']!=handle['epoch']:raise JournalError('stale_owner')
   if head['head']!=handle['revision']:raise JournalError('stale_revision')
   return self._append(head,copy.deepcopy(state),kind,head['epoch'],None if release else head['owner'],crash)
