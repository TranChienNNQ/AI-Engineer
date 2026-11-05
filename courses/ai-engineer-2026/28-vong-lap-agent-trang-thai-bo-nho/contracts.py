"""Small provider-neutral decision/state contract; no arbitrary code or writable tools."""
import hashlib
import json
import re
import unicodedata
from jsonschema import Draft202012Validator

VERSION='agent-loop-v1'

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def terms(text):return re.findall(r'\w+',unicodedata.normalize('NFC',text).casefold())
def units(obj):return len(terms(json.dumps(obj,ensure_ascii=False,sort_keys=True)))
def obj(props):return dict(type='object',properties=props,required=list(props),additionalProperties=False)
STR=dict(type='string',minLength=1,maxLength=100)
TOOLS={
 'get_ticket':obj(dict(ticket_id=STR)),
 'get_runbook':obj(dict(service=STR)),
 'get_health':obj(dict(service=STR))}
DECISION={'oneOf':[
    obj(dict(kind=dict(const='tool'),name=dict(const=name),args=args)) for name,args in TOOLS.items()
]+[obj(dict(kind=dict(const='final'),answer=dict(type='string',minLength=1,maxLength=800),source_ids=dict(type='array',items=STR,uniqueItems=True))),
   obj(dict(kind=dict(const='handoff'),reason=dict(enum=['outside_domain','unavailable','not_permitted','outside_task','ticket_required','need_human'])))]}
VALIDATOR=Draft202012Validator(DECISION);VALIDATOR.check_schema(DECISION)


def decision(raw):
    if not isinstance(raw,str) or len(raw.encode())>8000:raise ValueError('invalid_decision')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate_key')
            d[k]=v
        return d
    def bad_constant(value):raise ValueError('nonstandard_number')
    d=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad_constant)
    if not VALIDATOR.is_valid(d):raise ValueError('invalid_decision_schema')
    return d


def tool(name,**args):return dict(kind='tool',name=name,args=args)
TERMINAL={'DONE','HANDOFF','LIMIT_REACHED','STALLED','FAILED','CANCELLED'}
LIMITS=dict(max_turns=8,max_calls=6,max_context_units=6000,context_cap=900,output_reserve=64,max_repeats=2)
STATE_SCHEMA=obj(dict(
 schema_version=dict(const=VERSION),model_id=STR,run_id=STR,actor=STR,tenant=STR,
 task=obj(dict(ticket_id=STR,intent=dict(enum=['triage','retry']))),
 status=dict(enum=['READY','THINKING','PENDING_TOOL',*sorted(TERMINAL)]),
 turns=dict(type='integer',minimum=0),tool_reservations=dict(type='integer',minimum=0),
 context_units=dict(type='integer',minimum=0),limits=obj({k:dict(type='integer',minimum=1) for k in LIMITS}),
 observations=dict(type='object'),recent=dict(type='array',maxItems=2),repeats=dict(type='object'),
 pending=dict(type=['object','null']),answer=dict(type=['object','null']),reason=dict(type=['string','null']),
 revision=dict(type='integer',minimum=0)))
STATE_VALIDATOR=Draft202012Validator(STATE_SCHEMA);STATE_VALIDATOR.check_schema(STATE_SCHEMA)


def validate_state(s):
    if not STATE_VALIDATOR.is_valid(s):raise ValueError('invalid_checkpoint_schema')
