"""Synthetic ticket API. Effect + dedup receipt share ONE target-side transaction."""
import sqlite3,json
from contextlib import contextmanager
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from bridge53 import World48,Fault,canonical,digest
from contracts53 import Envelope,signature_valid
class Target:
 def __init__(self,path):
  self.db=sqlite3.connect(path,isolation_level=None,autocommit=sqlite3.LEGACY_TRANSACTION_CONTROL,timeout=2);self.db.row_factory=sqlite3.Row;self.db.execute('PRAGMA synchronous=FULL')
  self.db.executescript('''CREATE TABLE IF NOT EXISTS actors(token TEXT PRIMARY KEY,id TEXT UNIQUE,tenant TEXT,role TEXT,active INTEGER);
CREATE TABLE IF NOT EXISTS tickets(id INTEGER PRIMARY KEY,tenant TEXT,status TEXT,version INTEGER);
CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY,ticket_id INTEGER,tenant TEXT,actor TEXT,body TEXT,op_id TEXT,UNIQUE(tenant,actor,op_id));
CREATE TABLE IF NOT EXISTS receipts(tenant TEXT,actor TEXT,op_id TEXT,draft_hash TEXT,receipt TEXT,PRIMARY KEY(tenant,actor,op_id));
CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY,method TEXT,operation TEXT,actor TEXT);
''')
  if not self.db.execute('SELECT 1 FROM actors LIMIT 1').fetchone():
   w=World48(':memory:')
   try:self.db.executemany('INSERT INTO actors VALUES(?,?,?,?,?)',[(token,p['id'],p['tenant'],p['role'],int(p['active'])) for token,p in w.core.identities.items()])
   finally:w.close()
   self.db.executemany('INSERT INTO tickets VALUES(?,?,?,?)',[(1,'a','open',1),(2,'b','open',1)])
 def close(self):self.db.close()
 @contextmanager
 def tx(self):
  self.db.execute('BEGIN IMMEDIATE')
  try:yield;self.db.execute('COMMIT')
  except BaseException:
   if self.db.in_transaction:self.db.execute('ROLLBACK')
   raise
 def principal(self,token):
  p=self.db.execute('SELECT id,tenant,role,active FROM actors WHERE token=?',(token,)).fetchone()
  if not p or not p['active']:raise Fault(401,'unauthorized')
  return dict(p)|dict(active=True)
 def by_id(self,id_):
  row=self.db.execute('SELECT token FROM actors WHERE id=?',(id_,)).fetchone()
  if not row:raise Fault(401,'identity_not_found')
  return self.principal(row['token'])
 def ticket(self,p,id_):
  row=self.db.execute('SELECT * FROM tickets WHERE id=? AND tenant=?',(id_,p['tenant'])).fetchone()
  if not row:raise Fault(404,'target_not_found')
  return dict(row)
 def log(self,method,operation,p):self.db.execute('INSERT INTO requests(method,operation,actor) VALUES(?,?,?)',(method,operation,p['id']))
 def apply(self,token,envelope,now=100,before_commit=False):
  with self.tx():
   p=self.principal(token)
   if p['role']!='agent':raise Fault(403,'agent_required')
   e=Envelope.model_validate(envelope).model_dump();permit=e['permit'];draft=e['draft'];hash_=digest(draft)
   if not signature_valid(permit,e['signature']):raise Fault(403,'invalid_permit_signature')
   if permit['draft_hash']!=hash_ or permit['op_id']!=e['op_id'] or (permit['actor'],permit['tenant'])!=(p['id'],p['tenant']):raise Fault(403,'permit_binding_mismatch')
   reviewer=self.by_id(permit['reviewer'])
   if reviewer['role']!='reviewer' or reviewer['tenant']!=p['tenant'] or reviewer['id']==p['id']:raise Fault(403,'reviewer_unavailable')
   if now>=permit['expires_at'] or now>=draft['expires_at']:raise Fault(409,'permit_expired')
   ticket=self.ticket(p,draft['ticket_id'])
   old=self.db.execute('SELECT * FROM receipts WHERE tenant=? AND actor=? AND op_id=?',(p['tenant'],p['id'],e['op_id'])).fetchone()
   if old:
    if old['draft_hash']!=hash_:raise Fault(409,'idempotency_conflict')
    return json.loads(old['receipt'])
   if draft['expected_version']!=ticket['version']:raise Fault(409,'stale_target_version')
   if ticket['status']!='open':raise Fault(409,'target_closed')
   self.db.execute('INSERT INTO notes(ticket_id,tenant,actor,body,op_id) VALUES(?,?,?,?,?)',(ticket['id'],p['tenant'],p['id'],draft['body'],e['op_id']))
   self.db.execute('UPDATE tickets SET version=version+1 WHERE id=? AND version=?',(ticket['id'],ticket['version']))
   receipt=dict(op_id=e['op_id'],ticket_id=ticket['id'],version=ticket['version']+1,draft_hash=hash_,state='applied')
   self.db.execute('INSERT INTO receipts VALUES(?,?,?,?,?)',(p['tenant'],p['id'],e['op_id'],hash_,canonical(receipt)))
   if before_commit:raise Fault(503,'injected_before_target_commit')
   return receipt
 def status(self,token,op_id):
  p=self.principal(token)
  if p['role']!='agent':raise Fault(403,'agent_required')
  row=self.db.execute('SELECT receipt FROM receipts WHERE tenant=? AND actor=? AND op_id=?',(p['tenant'],p['id'],op_id)).fetchone()
  return dict(state='not_found') if row is None else dict(state='applied',receipt=json.loads(row['receipt']))

def build_app(path,now=100,fault=None):
 app=FastAPI()
 @app.middleware('http')
 async def errors(request,call_next):
  try:return await call_next(request)
  except Fault as e:return JSONResponse(dict(error=e.code),status_code=e.status)
 def token(request):return request.headers.get('authorization','').removeprefix('Bearer ')
 @app.get('/v1/tickets/{id_}')
 async def get_ticket(id_:int,request:Request):
  t=Target(path)
  try:p=t.principal(token(request));t.log('GET','ticket',p);return t.ticket(p,id_)
  finally:t.close()
 @app.post('/v1/notes')
 async def append(envelope:Envelope,request:Request):
  t=Target(path)
  try:
   p=t.principal(token(request));t.log('POST','append_internal_note',p)
   return t.apply(token(request),envelope.model_dump(),now,before_commit=fault=='before_commit')
  finally:t.close()
 @app.get('/v1/operations/{op_id}')
 async def status(op_id:str,request:Request):
  t=Target(path)
  try:
   p=t.principal(token(request));t.log('GET','operation_status',p)
   if fault=='status_unavailable':raise Fault(503,'status_unavailable')
   if fault=='status_missing':return dict(state='not_found') # Test-only visibility gap; not proof no effect.
   return t.status(token(request),op_id)
  finally:t.close()
 return app
