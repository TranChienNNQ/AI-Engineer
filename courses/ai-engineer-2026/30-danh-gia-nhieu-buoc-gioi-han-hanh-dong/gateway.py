"""Sequential trusted gateway with explicit modeled budgets, proposal binding and journal."""
from dataclasses import dataclass, asdict
import copy
import json
from domain import digest


class Rejected(Exception):
    pass


@dataclass(frozen=True)
class Limits:
    max_turns: int = 12
    max_calls: int = 8
    max_writes: int = 1
    max_units: int = 24
    deadline_ms: int = 900
    context_bytes: int = 6000

    def __post_init__(self):
        if any(type(v) is not int or v<0 for v in asdict(self).values()):
            raise ValueError('limits must be nonnegative integers')


TOOLS={'get_ticket':(1,40), 'get_runbook':(1,60), 'get_health':(1,50),
       'prepare_note':(1,20), 'commit_note':(3,80)}


def exact(obj, fields):
    if type(obj) is not dict or set(obj)!=set(fields):raise Rejected('invalid_fields')


def facts_valid(facts):
    exact(facts,['ticket_id','health','retry_limit'])
    if type(facts['ticket_id']) is not str or facts['ticket_id'] not in {'T-BLUE-1','T-GREEN-1'}:raise Rejected('invalid_ticket')
    if type(facts['health']) is not str or facts['health'] not in {'healthy','degraded'}:raise Rejected('invalid_health')
    if type(facts['retry_limit']) is not int or not 0<=facts['retry_limit']<=3:
        raise Rejected('invalid_retry_limit')


def parse_decision(raw):
    if not isinstance(raw,str) or len(raw.encode())>3000:raise Rejected('invalid_payload')
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise Rejected('duplicate_key')
            out[k]=v
        return out
    def bad_constant(value):raise Rejected('nonfinite')
    try:d=json.loads(raw,object_pairs_hook=unique,parse_constant=bad_constant)
    except (ValueError,RecursionError) as e:raise Rejected('invalid_json') from e
    if type(d) is not dict:raise Rejected('invalid_decision')
    if d.get('kind')=='tool':
        exact(d,['kind','name','args'])
        if type(d['name']) is not str or d['name'] not in TOOLS:raise Rejected('unknown_tool')
        a=d['args']
        if d['name'] in {'get_ticket','get_runbook','get_health'}:
            exact(a,['ticket_id'])
            if type(a['ticket_id']) is not str or a['ticket_id'] not in {'T-BLUE-1','T-GREEN-1'}:raise Rejected('invalid_ticket')
        elif d['name']=='prepare_note':
            exact(a,['facts']);facts_valid(a['facts'])
        else:
            exact(a,['draft_id','expected_version'])
            if type(a['draft_id']) is not str or len(a['draft_id'])>100:raise Rejected('invalid_draft')
            if type(a['expected_version']) is not int or a['expected_version']<1:raise Rejected('invalid_version')
    elif d.get('kind')=='final':
        exact(d,['kind','outcome','facts'])
        if type(d['outcome']) is not str or d['outcome'] not in {'NOTE','REPORT','HANDOFF'}:raise Rejected('invalid_outcome')
        if d['facts'] is not None:facts_valid(d['facts'])
    else:raise Rejected('invalid_kind')
    return d


class Gateway:
    def __init__(self,world,mode='strict',limits=None):
        if mode not in {'strict','schema_only'}:raise ValueError('unknown mode')
        self.world=world;self.case=world.case;self.mode=mode;self.limits=limits or Limits()
        self.turns=0;self.calls=0;self.writes_reserved=0;self.units=0;self.now_ms=0
        self.trace=[];self.observations=[];self.evidence=set();self.drafts={};self.approvals={}
        self.final=None;self.stop=None;self.cancelled=False;self.revoked=False

    def event(self,kind,**data):
        self.trace.append(dict(seq=len(self.trace)+1,kind=kind,at_ms=self.now_ms,**copy.deepcopy(data)))

    def reserve(self, *, turn=False, tool=None):
        cost=2 if turn else TOOLS[tool][0];ms=10 if turn else TOOLS[tool][1]
        if self.cancelled:raise Rejected('cancelled')
        if self.now_ms+ms>self.limits.deadline_ms:raise Rejected('deadline')
        if self.units+cost>self.limits.max_units:raise Rejected('unit_budget')
        if turn:
            if self.turns>=self.limits.max_turns:raise Rejected('turn_budget')
        else:
            if self.calls>=self.limits.max_calls:raise Rejected('call_budget')
            if tool=='commit_note' and self.writes_reserved>=self.limits.max_writes:raise Rejected('write_budget')
        self.units+=cost;self.now_ms+=ms
        if turn:self.turns+=1
        else:
            self.calls+=1
            if tool=='commit_note':self.writes_reserved+=1
        self.event('reserve',operation='model' if turn else tool,cost=cost)

    def view(self):
        public=dict(task=self.case.public(),observations=copy.deepcopy(self.observations))
        if len(json.dumps(public,ensure_ascii=False).encode())>self.limits.context_bytes:
            raise Rejected('context_budget')
        return public

    def model_turn(self):
        # Reserve BEFORE policy/model request. No refund on invalid output.
        public=self.view();self.reserve(turn=True)
        return public

    def call(self,name,args):
        # Teaching gateway is the sole exposed interface, sequential and host-trusted.
        self.reserve(tool=name)
        target=args.get('ticket_id','T-BLUE-1')
        if name=='prepare_note':target=args['facts']['ticket_id']
        if name=='commit_note':
            draft=self.drafts.get(args['draft_id'])
            if not draft:raise Rejected('draft_not_found')
            target=draft['facts']['ticket_id']
        if self.mode=='strict':
            if target!='T-BLUE-1' or not self.case.readable or self.revoked:raise Rejected('not_permitted')
            if name in {'prepare_note','commit_note'}:
                if self.case.intent!='NOTE':raise Rejected('read_only_intent')
                if not {'get_ticket','get_runbook','get_health'}<=self.evidence:raise Rejected('missing_evidence')
                health=next(o['data']['status'] for o in reversed(self.observations) if o['name']=='get_health' and 'error' not in o['data'])
                if health!='healthy':raise Rejected('unsafe_health')
        if name in {'get_ticket','get_runbook','get_health'}:
            data=self.world.read(name,target)
            if 'error' not in data and target=='T-BLUE-1':self.evidence.add(name)
        elif name=='prepare_note':
            version=self.world.tickets[target]['version'];facts=copy.deepcopy(args['facts'])
            ident='draft-'+digest(dict(task=self.case.case_id,version=version,facts=facts))[:16]
            draft=dict(task_id=self.case.case_id,ticket_id=target,version=version,facts=facts,
                       payload_hash=digest(facts),expires_ms=self.now_ms+250,consumed=False)
            self.drafts[ident]=draft
            # Trusted reviewer fixture, outside the policy/model interface. No real review UI.
            if self.case.review:
                self.approvals[ident]=copy.deepcopy(draft)
                self.event('review',draft_id=ident,payload_hash=draft['payload_hash'])
            data=dict(draft_id=ident,expected_version=version,reviewed=ident in self.approvals)
        else:
            approval=self.approvals.get(args['draft_id'])
            if self.mode=='strict':
                if not approval:raise Rejected('review_required')
                if draft['consumed']:raise Rejected('review_consumed')
                if self.now_ms>=approval['expires_ms']:raise Rejected('review_expired')
                if draft['payload_hash']!=digest(draft['facts']) or approval!=draft:raise Rejected('review_mismatch')
                if args['expected_version']!=draft['version'] or self.world.tickets[target]['version']!=draft['version']:
                    raise Rejected('stale_version')
            data=self.world.write(target,draft['facts'],approved=approval,
                                  evidence=self.evidence,now=self.now_ms)
            draft['consumed']=True
        self.event('tool_result',name=name,target=target,error=data.get('error'))
        self.observations.append(dict(name=name,data=copy.deepcopy(data),authority='data'))
        return data

    def execute(self,decision):
        if decision['kind']=='final':
            self.final=copy.deepcopy(decision);self.stop='FINAL';self.event('final',outcome=decision['outcome']);return
        name=decision['name'];args=decision['args'];self.event('proposal',name=name,args=args)
        try:return self.call(name,args)
        except Rejected as exc:
            code=str(exc);self.event('denied',name=name,code=code)
            self.observations.append(dict(name=name,data=dict(error=code),authority='data'))
            if code in {'cancelled','deadline','unit_budget','turn_budget','call_budget','write_budget'}:
                self.stop='LIMIT' if code!='cancelled' else 'CANCELLED'
            return dict(error=code)

    def counters(self):
        return dict(turns=self.turns,calls=self.calls,writes_reserved=self.writes_reserved,
                    units=self.units,modeled_ms=self.now_ms)
