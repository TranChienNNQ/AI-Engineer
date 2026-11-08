import hashlib,json
from pathlib import Path
from bridge import digest,Fault
from fixtures import CASES,task,mock,terminal
from tools import World
from runtime import Agent
from agent_contract import Terminal

def execute(out,case):
    w=World(Path(out)/(case['id']+'.sqlite'),health_status=case.get('health_status','degraded'),health_down=case.get('health_down',False),search_down=case.get('search_down',False))
    adapter,script,clock=mock(case['profile']);agent=Agent(w,adapter,max_tools=case.get('max_tools',3),max_turns=case.get('max_turns',6),clock=clock)
    try:
        r=agent.run(case.get('actor','demo-agent-a'),task(goal=case.get('goal','triage'),question=case.get('question','gateway timeout 504 upstream trace_id'),ticket_id=case.get('ticket_id',1),service=case.get('service','gateway')))
        return dict(id=case['id'],expected=dict(state=case['state'],code=case['code'],reads=case['reads']),correct=(r['state'],r['code'],r['actual_reads'])==(case['state'],case['code'],case['reads']),http_attempts=len(script.requests),result=r,manifest=agent.manifest)
    finally:agent.close()

def baseline(out,case):
    # Fixed goal-dependent graph; no model call. Same World, final gate, task and read semantics.
    w=World(Path(out)/(case['id']+'-baseline.sqlite'),health_status=case.get('health_status','degraded'))
    adapter,script,clock=mock();a=Agent(w,adapter,clock=clock);obs=[];t=task(goal=case.get('goal','triage'),question=case.get('question','gateway timeout 504 upstream trace_id'))
    try:
        t=w.bind('demo-agent-a',t)
        actions=[('read_ticket',dict(ticket_id=1)),('search_runbooks',dict(query=t['question']))]
        if t['goal']=='triage':actions.append(('read_health',{}))
        final=None
        for n,(name,args) in enumerate(actions,1):
            o=dict(id=f'O{n}',call_id=f'baseline_{n}',tool=name,args=args,**w.dispatch(name,args,t,'demo-agent-a'));obs.append(o)
            if not o['ok']:final=terminal(obs,'handoff','tool_unavailable');break
            if name=='search_runbooks' and not o['data']['evidence']:final=terminal(obs,'handoff','need_info');break
            if name=='read_health' and o['data']['status']=='unknown':final=terminal(obs,'handoff','need_info');break
        result=a.finish(Terminal.model_validate(final or terminal(obs)),t,'demo-agent-a',obs)
        return result|dict(actual_reads=w.actual_reads,model_calls=0,effects=w.effects())
    finally:a.close()

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);rows=[execute(out,c) for c in CASES];byid={r['id']:r for r in rows}
    compare=[]
    for name in ['triage','retry_only','health_unknown','no_source']:
        c=next(c for c in CASES if c['id']==name);base=baseline(out,c);agent=byid[name]['result']
        compare.append(dict(id=name,agent_state=agent['state'],baseline_state=base['state'],same_outcome=(agent['state'],agent['code'])==(base['state'],base['code']),agent_reads=agent['actual_reads'],baseline_reads=base['actual_reads'],agent_model_calls=agent['model_calls'],baseline_model_calls=0))
    manifest=dict(case_snapshot=digest(CASES),default_release=rows[0]['result']['release'],implementation=rows[0]['manifest']['implementation'],base_source=rows[0]['manifest']['base_source'],fixture_code=hashlib.sha256((Path(__file__).parent/'fixtures.py').read_bytes()).hexdigest(),experiment_code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='scripted protocol/branch tests, not model evaluation')
    result=dict(cases=len(rows),correct=sum(r['correct'] for r in rows),rows=rows,comparison=compare,manifest=manifest,experiment_id=digest(manifest)[:16],
        model_calls=sum(r['result']['model_calls'] for r in rows),http_attempts=sum(r['http_attempts'] for r in rows),
        known_input_tokens=sum(r['result']['known_input_tokens'] for r in rows),known_output_tokens=sum(r['result']['known_output_tokens'] for r in rows),unknown_usage_attempts=sum(r['result']['unknown_usage_attempts'] for r in rows),live_model_calls=0)
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result
