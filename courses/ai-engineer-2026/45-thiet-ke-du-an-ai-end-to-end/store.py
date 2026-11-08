"""Single SQLite failure domain: effects and receipts commit together. No remote ticket API."""
from contextlib import contextmanager
import json,sqlite3,threading
from pathlib import Path
from corpus import digest


class Fault(Exception):
    def __init__(self,status,code):self.status=status;self.code=code;super().__init__(code)


class Store:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock()
        self.db=sqlite3.connect(self.path,isolation_level=None,check_same_thread=False,timeout=1.)
        self.db.row_factory=sqlite3.Row;self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY,tenant TEXT,actor TEXT,request_key TEXT,request_hash TEXT,result TEXT,UNIQUE(tenant,actor,request_key));
        CREATE TABLE IF NOT EXISTS proposals(id INTEGER PRIMARY KEY,run_id INTEGER UNIQUE,tenant TEXT,actor TEXT,payload TEXT,payload_hash TEXT,state TEXT,reviewer TEXT,expires REAL,FOREIGN KEY(run_id) REFERENCES runs(id));
        CREATE TABLE IF NOT EXISTS tickets(id INTEGER PRIMARY KEY,tenant TEXT,title TEXT,body TEXT,status TEXT,version INTEGER);
        CREATE TABLE IF NOT EXISTS receipts(tenant TEXT,actor TEXT,request_key TEXT,proposal_id INTEGER,result TEXT,PRIMARY KEY(tenant,actor,request_key),UNIQUE(proposal_id));
        CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY,kind TEXT,entity INTEGER,actor TEXT,code TEXT);
        ''')
        if not self.db.execute('SELECT 1 FROM tickets LIMIT 1').fetchone():
            self.db.executemany('INSERT INTO tickets VALUES(?,?,?,?,?,?)',[(1,'a','Gateway slow','Fixture ticket A','open',1),(2,'b','Database lag','Fixture ticket B','open',1)])

    def close(self):self.db.close()

    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:yield;self.db.execute('COMMIT')
            except BaseException:
                if self.db.in_transaction:self.db.execute('ROLLBACK')
                raise

    def ticket(self,id_,principal):
        row=self.db.execute('SELECT * FROM tickets WHERE id=? AND tenant=?',(id_,principal['tenant'])).fetchone()
        if not row:raise Fault(404,'ticket_not_found')
        return dict(row)

    def begin(self,p,key,hash_):
        with self.transaction():
            old=self.db.execute('SELECT * FROM runs WHERE tenant=? AND actor=? AND request_key=?',(p['tenant'],p['id'],key)).fetchone()
            if old:
                if old['request_hash']!=hash_:raise Fault(409,'request_key_conflict')
                if old['result'] is None:raise Fault(409,'run_in_progress')
                return old['id'],json.loads(old['result'])
            cur=self.db.execute('INSERT INTO runs(tenant,actor,request_key,request_hash) VALUES(?,?,?,?)',(p['tenant'],p['id'],key,hash_))
            return cur.lastrowid,None

    def finish(self,id_,result,payload=None):
        with self.transaction():
            if payload is not None:
                row=self.db.execute('SELECT * FROM runs WHERE id=?',(id_,)).fetchone()
                cur=self.db.execute('INSERT INTO proposals(run_id,tenant,actor,payload,payload_hash,state) VALUES(?,?,?,?,?,?)',(id_,row['tenant'],row['actor'],json.dumps(payload,ensure_ascii=False,sort_keys=True),digest(payload),'pending'))
                result['proposal_id']=cur.lastrowid
            result['run_id']=id_;self.db.execute('UPDATE runs SET result=? WHERE id=?',(json.dumps(result,ensure_ascii=False,sort_keys=True),id_))
            self.db.execute('INSERT INTO audit(kind,entity,actor,code) SELECT ?,id,actor,? FROM runs WHERE id=?',('run',result['state'],id_))
        return result

    def get_run(self,id_,p):
        row=self.db.execute('SELECT result FROM runs WHERE id=? AND tenant=? AND actor=?',(id_,p['tenant'],p['id'])).fetchone()
        if not row:raise Fault(404,'run_not_found')
        if row['result'] is None:raise Fault(409,'run_in_progress')
        return json.loads(row['result'])

    def proposal(self,id_,p,review=False):
        row=self.db.execute('SELECT * FROM proposals WHERE id=? AND tenant=?',(id_,p['tenant'])).fetchone()
        if not row or not review and row['actor']!=p['id']:raise Fault(404,'proposal_not_found')
        out=dict(row);out['payload']=json.loads(out['payload']);return out

    def approve(self,id_,p,now):
        if p['role']!='reviewer':raise Fault(403,'reviewer_required')
        with self.transaction():
            row=self.proposal(id_,p,review=True)
            if row['actor']==p['id']:raise Fault(403,'self_review_denied')
            if row['payload_hash']!=digest(row['payload']):raise Fault(409,'proposal_digest_mismatch')
            if row['state']!='pending':raise Fault(409,'proposal_not_pending')
            self.db.execute('UPDATE proposals SET state=?,reviewer=?,expires=? WHERE id=?',('approved',p['id'],now+60,id_))
            self.db.execute('INSERT INTO audit(kind,entity,actor,code) VALUES(?,?,?,?)',('review',id_,p['id'],'approved'))
        return dict(proposal_id=id_,state='approved',payload_hash=row['payload_hash'],expires=now+60,payload=row['payload'])

    def execute(self,id_,p,key,now,release,validate_payload,fault=None):
        if p['role']!='agent':raise Fault(403,'agent_required')
        with self.transaction():
            proposal=self.proposal(id_,p);payload=proposal['payload']
            old=self.db.execute('SELECT * FROM receipts WHERE tenant=? AND actor=? AND request_key=?',(p['tenant'],p['id'],key)).fetchone()
            if old:
                if old['proposal_id']!=id_:raise Fault(409,'receipt_key_conflict')
                return json.loads(old['result'])
            prior=self.db.execute('SELECT 1 FROM receipts WHERE proposal_id=?',(id_,)).fetchone()
            if prior:raise Fault(409,'proposal_already_executed')
            if proposal['payload_hash']!=digest(payload):raise Fault(409,'proposal_digest_mismatch')
            if proposal['state']!='approved':raise Fault(409,'approval_required')
            if proposal['expires']<=now:raise Fault(409,'approval_expired')
            if payload['release']!=release:raise Fault(409,'release_changed')
            validate_payload(payload,p)
            op=payload['operation'];ticket_id=payload['ticket_id']
            if op=='create':
                cur=self.db.execute('INSERT INTO tickets(tenant,title,body,status,version) VALUES(?,?,?,?,?)',(p['tenant'],payload['title'],payload['body'],'open',1));ticket_id=cur.lastrowid
            else:
                t=self.ticket(ticket_id,p)
                if t['version']!=payload['expected_version']:raise Fault(409,'stale_ticket_version')
                if t['status']!='open':raise Fault(409,'ticket_closed')
                if op=='note':self.db.execute('UPDATE tickets SET body=body||?,version=version+1 WHERE id=?',('\n'+payload['body'],ticket_id))
                elif op=='resolve':self.db.execute('UPDATE tickets SET status=?,version=version+1 WHERE id=?',('resolved',ticket_id))
                else:raise Fault(422,'operation_not_allowed')
            if fault=='before_commit':raise Fault(503,'injected_before_commit')
            t=self.ticket(ticket_id,p);result=dict(proposal_id=id_,ticket_id=ticket_id,version=t['version'],status=t['status'],release=release,state='committed')
            self.db.execute('INSERT INTO receipts VALUES(?,?,?,?,?)',(p['tenant'],p['id'],key,id_,json.dumps(result,sort_keys=True)))
            self.db.execute('UPDATE proposals SET state=? WHERE id=?',('executed',id_))
            self.db.execute('INSERT INTO audit(kind,entity,actor,code) VALUES(?,?,?,?)',('effect',id_,p['id'],op))
        if fault=='after_commit':raise Fault(503,'injected_lost_response')
        return result
