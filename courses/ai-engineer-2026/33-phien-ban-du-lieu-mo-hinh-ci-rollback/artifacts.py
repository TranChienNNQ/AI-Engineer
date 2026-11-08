"""Local content-addressed object store and registry in a single SQLite database."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


class Registry:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,isolation_level=None,timeout=5)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, body BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS releases (id TEXT PRIMARY KEY, manifest TEXT NOT NULL REFERENCES objects(id));
          CREATE TABLE IF NOT EXISTS approvals (release TEXT PRIMARY KEY REFERENCES releases(id), baseline TEXT NOT NULL REFERENCES releases(id), policy TEXT NOT NULL, report TEXT NOT NULL REFERENCES objects(id));
          CREATE TABLE IF NOT EXISTS deployment (slot TEXT PRIMARY KEY, generation INTEGER NOT NULL, release TEXT REFERENCES releases(id));
          INSERT OR IGNORE INTO deployment VALUES ('production',0,NULL);
          CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, old_release TEXT, new_release TEXT NOT NULL, generation INTEGER NOT NULL, reason TEXT NOT NULL);
        ''')

    def close(self):self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def put(self,value):
        body=canonical(value);key=hashlib.sha256(body).hexdigest()
        self.db.execute('INSERT OR IGNORE INTO objects VALUES (?,?)',(key,body))
        if self.db.execute('SELECT body FROM objects WHERE id=?',(key,)).fetchone()[0]!=body:raise ValueError('object_corrupt')
        return key

    def get(self,key):
        if not isinstance(key,str) or not re.fullmatch('[0-9a-f]{64}',key):raise ValueError('digest_format')
        row=self.db.execute('SELECT body FROM objects WHERE id=?',(key,)).fetchone()
        if row is None:raise ValueError('object_missing')
        if hashlib.sha256(row[0]).hexdigest()!=key:raise ValueError('digest_mismatch')
        return json.loads(row[0])

    def register(self,release,manifest):
        if not re.fullmatch('[a-z][a-z0-9-]{0,47}',release):raise ValueError('release_name')
        key=self.put(manifest)
        old=self.db.execute('SELECT manifest FROM releases WHERE id=?',(release,)).fetchone()
        if old and old[0]!=key:raise ValueError('release_is_immutable')
        self.db.execute('INSERT OR IGNORE INTO releases VALUES (?,?)',(release,key))
        return key

    def manifest(self,release):
        row=self.db.execute('SELECT manifest FROM releases WHERE id=?',(release,)).fetchone()
        if row is None:raise ValueError('release_missing')
        return self.get(row[0])

    def state(self):
        generation,release=self.db.execute("SELECT generation,release FROM deployment WHERE slot='production'").fetchone()
        return dict(generation=generation,release=release)

    def history(self):
        return [dict(old=a,new=b,generation=c,reason=d) for a,b,c,d in self.db.execute('SELECT old_release,new_release,generation,reason FROM events ORDER BY seq')]
