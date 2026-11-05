"""In-memory authorization and action gateway. Not an authentication server or sandbox."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import secrets
from threading import RLock

from jsonschema import Draft202012Validator


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


@dataclass
class Principal:
    id:str
    tenant:str
    roles:tuple
    scopes:tuple
    clearance:int
    active:bool=True
    revision:int=1


@dataclass
class Resource:
    id:str
    tenant:str|None
    kind:str
    level:int
    audience:tuple
    text:str
    version:int=1
    status:str='open'


@dataclass
class Record:
    id:str
    owner:str
    kind:str
    text:str
    origins:list
    version:int=1


@dataclass
class World:
    principals:dict
    sessions:dict
    resources:dict
    records:dict=field(default_factory=dict)
    denied:set=field(default_factory=set)
    acl_revision:int=1
    corpus_revision:int=1
    memory_revision:int=1
    lock:RLock=field(default_factory=RLock,repr=False)

    def resolve(self,session):
        uid=self.sessions.get(session)
        p=self.principals.get(uid)
        if p is None or not p.active:
            raise PermissionError('not_authenticated')
        return p

    def fingerprint(self,p):
        return digest(asdict(p))

    def can_read(self,p,r):
        if r is None or not p.active or (p.id,r.id) in self.denied:
            return False
        scope={'document':'kb:read','ticket':'tickets:read'}.get(r.kind)
        tenant_ok=(r.tenant==p.tenant or (r.tenant is None and r.level==0))
        return bool(scope in p.scopes and tenant_ok and r.level<=p.clearance
                    and set(p.roles)&set(r.audience))

    def can_close(self,p,r):
        return (self.can_read(p,r) and r.kind=='ticket' and r.tenant==p.tenant
                and 'operator' in p.roles and 'tickets:close' in p.scopes)

    def ref(self,r):
        return dict(id=r.id,version=r.version,body_hash=digest([r.text,r.status]))

    def valid_ref(self,p,ref):
        r=self.resources.get(ref['id'])
        return self.can_read(p,r) and self.ref(r)==ref

    def record_allowed(self,p,record):
        return (record.owner==p.id and record.kind in {'history','memory','tool_result'}
                and bool(record.origins) and all(self.valid_ref(p,ref) for ref in record.origins))

    def revoke(self,uid,resource_id):
        with self.lock:
            self.denied.add((uid,resource_id));self.acl_revision+=1

    def update_resource(self,resource_id,text):
        with self.lock:
            r=self.resources[resource_id];r.text=text;r.version+=1;self.corpus_revision+=1

    def deactivate(self,uid):
        with self.lock:
            self.principals[uid].active=False
            self.principals[uid].revision+=1;self.acl_revision+=1


ID=dict(type='string',pattern=r'^T-[A-Z]+-[0-9]+$')
READ=dict(type='object',additionalProperties=False,required=['tool','args'],properties={
    'tool':dict(const='get_ticket'),
    'args':dict(type='object',additionalProperties=False,required=['ticket_id'],properties={'ticket_id':ID})})
CLOSE=dict(type='object',additionalProperties=False,required=['tool','args'],properties={
    'tool':dict(const='close_ticket'),
    'args':dict(type='object',additionalProperties=False,required=['ticket_id','expected_version','reason'],properties={
        'ticket_id':ID,'expected_version':dict(type='integer',minimum=1),
        'reason':dict(type='string',minLength=1,maxLength=200)})})
SCHEMA={'$schema':'https://json-schema.org/draft/2020-12/schema','oneOf':[READ,CLOSE]}
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR=Draft202012Validator(SCHEMA)


def parse_proposal(raw):
    if not isinstance(raw,str) or len(raw.encode())>8000:
        raise ValueError('invalid_proposal')
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate_key')
            result[k]=v
        return result
    def bad_constant(value):raise ValueError('nonstandard_number')
    obj=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad_constant)
    if not VALIDATOR.is_valid(obj):raise ValueError('invalid_schema')
    if obj['tool']=='close_ticket' and not obj['args']['reason'].strip():raise ValueError('empty_reason')
    return obj


class Gateway:
    def __init__(self,world):
        self.world=world;self.grants={};self.audit=[]

    def issue_after_review(self,session,reviewed_raw,now=0,ttl=30):
        """TRUSTED controller only; not a model tool. Review UI/authentication is outside this lab."""
        if type(ttl) is not int or ttl<=0:raise ValueError('positive TTL required')
        obj=parse_proposal(reviewed_raw)
        with self.world.lock:
            p=self.world.resolve(session);r=self.world.resources.get(obj['args']['ticket_id'])
            if obj['tool']!='close_ticket' or not self.world.can_close(p,r):raise PermissionError('not_permitted')
            if r.status!='open' or r.version!=obj['args']['expected_version']:raise ValueError('resource_changed')
            token=secrets.token_urlsafe(24)
            self.grants[token]=dict(subject=p.id,principal_revision=p.revision,tenant=p.tenant,
                                    proposal_hash=digest(obj),resource_ref=self.world.ref(r),acl_revision=self.world.acl_revision,
                                    expires_at=now+ttl,used=False)
            return token

    def execute(self,session,raw,now=0,grant=None):
        # One lock covers live authorization, grant consumption and in-memory state mutation.
        with self.world.lock:
            try:p=self.world.resolve(session)
            except PermissionError:
                self.audit.append(dict(decision='deny',reason='not_authenticated'))
                return dict(decision='deny',reason='not_authenticated',effect=False)
            try:obj=parse_proposal(raw)
            except (ValueError,TypeError):
                self.audit.append(dict(subject=p.id,decision='deny',reason='invalid_proposal',proposal_hash=digest(raw if isinstance(raw,str) else {'invalid_type':type(raw).__name__})))
                return dict(decision='deny',reason='invalid_proposal',effect=False)
            tool=obj['tool'];args=obj['args'];r=self.world.resources.get(args['ticket_id'])
            def finish(allowed,reason,effect=False,data=None):
                # No raw payload, grant bearer token, source body or credential in audit.
                self.audit.append(dict(subject=p.id,tenant=p.tenant,tool=tool,resource_id=args['ticket_id'],
                                       decision='allow' if allowed else 'deny',reason=reason,
                                       acl_revision=self.world.acl_revision,proposal_hash=digest(obj)))
                return dict(decision='allow' if allowed else 'deny',reason=reason,effect=effect,data=data)
            if tool=='get_ticket':
                if r is None or r.kind!='ticket' or not self.world.can_read(p,r):return finish(False,'not_permitted')
                return finish(True,'ok',data=dict(ticket_id=r.id,status=r.status,text=r.text,origin=self.world.ref(r)))
            if not self.world.can_close(p,r):return finish(False,'not_permitted')
            g=self.grants.get(grant) if isinstance(grant,str) else None
            if g is None:return finish(False,'grant_required')
            if (g['used'] or now>=g['expires_at'] or g['subject']!=p.id or g['tenant']!=p.tenant
                    or g['principal_revision']!=p.revision or g['proposal_hash']!=digest(obj)
                    or g['acl_revision']!=self.world.acl_revision):
                return finish(False,'grant_invalid')
            if g['resource_ref']!=self.world.ref(r) or args['expected_version']!=r.version or r.status!='open':
                return finish(False,'resource_changed')
            g['used']=True;r.status='closed';r.version+=1;self.world.corpus_revision+=1
            return finish(True,'ok',effect=True,data=dict(ticket_id=r.id,status=r.status,version=r.version))
