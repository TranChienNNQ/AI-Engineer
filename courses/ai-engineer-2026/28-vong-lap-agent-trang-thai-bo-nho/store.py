"""File-backed checkpoints/events and reviewed memory. DB file/host code are trusted, not a sandbox."""
import copy
import json
import sqlite3
from contracts import VERSION,LIMITS,validate_state,digest,terms


class Conflict(Exception):pass


class Store:
    def __init__(self,path):
        # Explicit SQL transactions. SQLite's autocommit here is intentional.
        self.db=sqlite3.connect(str(path),autocommit=True,timeout=5)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS runs(
            id TEXT PRIMARY KEY,actor TEXT NOT NULL,tenant TEXT NOT NULL,revision INTEGER NOT NULL,state TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
            run_id TEXT NOT NULL REFERENCES runs(id),seq INTEGER NOT NULL,event TEXT NOT NULL,PRIMARY KEY(run_id,seq));
        CREATE TABLE IF NOT EXISTS memory(
            tenant TEXT NOT NULL,actor TEXT NOT NULL,key TEXT NOT NULL,version INTEGER NOT NULL,
            body TEXT NOT NULL,refs TEXT NOT NULL,expires INTEGER NOT NULL,status TEXT NOT NULL,
            PRIMARY KEY(tenant,actor,key));''')

    def close(self):self.db.close()

    def create(self,run_id,actor,world,ticket_id='T-BLUE-1',intent='triage',model_id='adaptive-v1',**limits):
        p=world.principal(actor)
        if set(limits)-set(LIMITS):raise ValueError('unknown_limit')
        cfg=LIMITS|limits
        s=dict(schema_version=VERSION,model_id=model_id,run_id=run_id,actor=actor,tenant=p['tenant'],
               task=dict(ticket_id=ticket_id,intent=intent),status='READY',turns=0,tool_reservations=0,
               context_units=0,limits=cfg,observations={},recent=[],repeats={},pending=None,answer=None,reason=None,revision=0)
        validate_state(s)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('INSERT INTO runs VALUES(?,?,?,?,?)',(run_id,actor,p['tenant'],0,json.dumps(s,ensure_ascii=False)))
            self.db.execute('INSERT INTO events VALUES(?,?,?)',(run_id,0,json.dumps(dict(type='created',status='READY'))))
            self.db.execute('COMMIT')
        except BaseException:self.db.execute('ROLLBACK');raise
        return s

    def load(self,run_id,actor,world):
        p=world.principal(actor)
        row=self.db.execute('SELECT revision,state FROM runs WHERE id=? AND actor=? AND tenant=?',(run_id,actor,p['tenant'])).fetchone()
        if not row:raise PermissionError('run_not_accessible')
        s=json.loads(row[1]);validate_state(s)
        if (s['run_id'],s['actor'],s['tenant'],s['revision'])!=(run_id,actor,p['tenant'],row[0]):raise ValueError('checkpoint_identity_mismatch')
        return s

    def save(self,s,event):
        validate_state(s);new=copy.deepcopy(s);old=s['revision'];new['revision']=old+1
        self.db.execute('BEGIN IMMEDIATE')
        try:
            updated=self.db.execute('UPDATE runs SET revision=?,state=? WHERE id=? AND actor=? AND tenant=? AND revision=?',
                (new['revision'],json.dumps(new,ensure_ascii=False),s['run_id'],s['actor'],s['tenant'],old))
            if updated.rowcount!=1:raise Conflict('stale_checkpoint')
            self.db.execute('INSERT INTO events VALUES(?,?,?)',(s['run_id'],new['revision'],json.dumps(event,ensure_ascii=False)))
            self.db.execute('COMMIT')
        except BaseException:self.db.execute('ROLLBACK');raise
        return new

    def events(self,run_id,actor,world):
        self.load(run_id,actor,world)
        return [json.loads(row[0]) for row in self.db.execute('SELECT event FROM events WHERE run_id=? ORDER BY seq',(run_id,))]

    def remember_reviewed(self,actor,world,key,text,refs,*,now=0,ttl=100,expected_version=0):
        """Trusted controller only: explicit review is assumed; never a model tool."""
        p=world.principal(actor)
        if not isinstance(key,str) or not 1<=len(key)<=80 or not isinstance(text,str) or not 1<=len(text)<=500:raise ValueError('invalid_memory')
        if type(ttl) is not int or ttl<1:raise ValueError('invalid_ttl')
        if type(expected_version) is not int or expected_version<0:raise ValueError('invalid_memory_version')
        if not world.valid_refs(actor,refs):raise PermissionError('invalid_memory_origins')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.db.execute('SELECT version FROM memory WHERE tenant=? AND actor=? AND key=?',(p['tenant'],actor,key)).fetchone()
            version=row[0] if row else 0
            if version!=expected_version:raise Conflict('stale_memory_version')
            self.db.execute('INSERT INTO memory VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(tenant,actor,key) DO UPDATE SET version=excluded.version,body=excluded.body,refs=excluded.refs,expires=excluded.expires,status=excluded.status',
                (p['tenant'],actor,key,version+1,text,json.dumps(refs,ensure_ascii=False),now+ttl,'active'))
            self.db.execute('COMMIT')
        except BaseException:self.db.execute('ROLLBACK');raise
        return version+1

    def recall(self,actor,world,query,*,now=0,k=2):
        p=world.principal(actor)
        if type(k) is not int or not 1<=k<=10:raise ValueError('invalid_k')
        q=set(terms(query));candidates=[]
        rows=self.db.execute('SELECT key,version,body,refs,expires FROM memory WHERE tenant=? AND actor=? AND status=? AND expires>?',
            (p['tenant'],actor,'active',now)).fetchall()
        for key,version,text,raw_refs,expiry in rows:
            refs=json.loads(raw_refs)
            if not world.valid_refs(actor,refs):continue
            score=len(q&set(terms(text)))
            if score:candidates.append((score,key,dict(key=key,version=version,text=text,origins=refs,expires=expiry,authority='data')))
        return [item for _,_,item in sorted(candidates,key=lambda x:(-x[0],x[1]))[:k]]

    def forget(self,actor,world,key,expected_version):
        if type(expected_version) is not int or expected_version<1:raise ValueError('invalid_memory_version')
        p=world.principal(actor)
        row=self.db.execute('UPDATE memory SET status=?,version=version+1 WHERE tenant=? AND actor=? AND key=? AND version=?',
                            ('deleted',p['tenant'],actor,key,expected_version))
        if row.rowcount!=1:raise Conflict('stale_memory_version')
