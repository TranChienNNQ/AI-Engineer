"""Provider-neutral teaching protocol. These are NOT fields of a particular LLM API."""
import hashlib
import json
from jsonschema import Draft202012Validator


def digest(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def object_schema(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)

STR=dict(type='string',minLength=1,maxLength=80)
TICKET=dict(type='string',pattern=r'^T-[A-Z]+-[0-9]+$')
ARGS={
    'get_ticket':object_schema(dict(ticket_id=TICKET)),
    'lookup_runbook':object_schema(dict(service=STR)),
    'get_health':object_schema(dict(service=STR)),
    'append_note':object_schema(dict(ticket_id=TICKET,expected_version=dict(type='integer',minimum=1),
                                     note=dict(type='string',minLength=1,maxLength=600))),
}
BRANCHES=[object_schema(dict(id=STR,name=dict(const=name),arguments=schema)) for name,schema in ARGS.items()]
TURN_SCHEMA=dict(type='array',minItems=1,maxItems=4,items=dict(oneOf=BRANCHES))
VALIDATOR=Draft202012Validator(TURN_SCHEMA);VALIDATOR.check_schema(TURN_SCHEMA)


def parse_turn(raw):
    if not isinstance(raw,str) or len(raw.encode())>16000:raise ValueError('invalid_turn')
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate_key')
            out[k]=v
        return out
    def bad_constant(value):raise ValueError('nonstandard_number')
    calls=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad_constant)
    if not VALIDATOR.is_valid(calls):raise ValueError('invalid_schema')
    if len({c['id'] for c in calls})!=len(calls):raise ValueError('duplicate_call_id')
    if any(c['name']=='append_note' and not c['arguments']['note'].strip() for c in calls):raise ValueError('blank_note')
    return calls


def call(cid,name,**arguments):
    return parse_turn(json.dumps([dict(id=cid,name=name,arguments=arguments)],ensure_ascii=False))[0]


def validate_output(name,data):
    schemas={
      'get_ticket':object_schema(dict(ticket_id=TICKET,service=STR,category=STR,version=dict(type='integer',minimum=1))),
      'lookup_runbook':object_schema(dict(sources=dict(type='array',items=object_schema(dict(id=STR,version=dict(type='integer',minimum=1),text=dict(type='string',maxLength=2000)))))),
      'get_health':object_schema(dict(service=STR,state=dict(enum=['healthy','degraded','unknown']))),
      'append_note':object_schema(dict(ticket_id=TICKET,note_id=STR,version=dict(type='integer',minimum=2))),
    }
    return name in schemas and Draft202012Validator(schemas[name]).is_valid(data)


def pair_results(calls,results):
    """Validate full result set; result completion order may differ from call order."""
    expected={c['id']:c for c in calls}
    if len(expected)!=len(calls) or len(results)!=len(calls):raise ValueError('correlation_mismatch')
    for r in results:
        if not isinstance(r,dict) or set(r)!={'call_id','name','status','data','error'}:raise ValueError('invalid_result_envelope')
        if not isinstance(r['call_id'],str):raise ValueError('invalid_result_id')
        if r['status']=='ok' and r['error'] is not None:raise ValueError('invalid_result_error')
        if r['status']=='error' and (not isinstance(r['error'],str) or not r['error'] or r['data'] is not None):raise ValueError('invalid_result_error')
    by_id={r['call_id']:r for r in results}
    if len(by_id)!=len(results) or set(by_id)!=set(expected):raise ValueError('correlation_mismatch')
    for cid,r in by_id.items():
        if r['name']!=expected[cid]['name']:raise ValueError('wrong_result_tool')
        if r['status'] not in {'ok','error'}:raise ValueError('wrong_result_status')
        if r['status']=='ok' and not validate_output(r['name'],r['data']):raise ValueError('invalid_output')
    return [by_id[c['id']] for c in calls]
