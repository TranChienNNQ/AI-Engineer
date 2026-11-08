import json,hashlib
from pathlib import Path
from bridge50 import World48,Link,MCPWorld,MCPAgent,digest,task,source_hash
from bridge50 import b as previous
from planning import PlanningPort,PlannedAgent,implementation_hash

TIME='2026-11-08T02:00:00Z'
CASES=[
 dict(id='straight_memory',transport='memory',state='answered',code='ok',sent=3,revisions=1),
 dict(id='straight_stdio',transport='stdio',state='answered',code='ok',sent=3,revisions=1),
 dict(id='recover_memory',transport='memory',ambiguous=True,state='answered',code='ok',sent=4,revisions=2),
 dict(id='recover_stdio',transport='stdio',ambiguous=True,state='answered',code='ok',sent=4,revisions=2),
 dict(id='discrepancy',transport='memory',health_status='ok',state='handoff',code='uncertain',sent=3,revisions=2),
 dict(id='different_time',transport='memory',health_status='ok',report_at='2026-11-07T02:00:00Z',state='answered',code='ok',sent=3,revisions=1),
 dict(id='unknown_health',transport='memory',health_status='unknown',state='handoff',code='need_info',sent=3,revisions=2),
 dict(id='tool_failure',transport='memory',fault='tool_error',state='handoff',code='tool_unavailable',sent=3,revisions=2),
 dict(id='no_new_hint',transport='memory',ambiguous=True,symptom='quantum cooling',state='handoff',code='need_info',sent=3,revisions=3),
 dict(id='cycle',transport='memory',profile='cycle',state='failed',code='plan_cyclic_plan',sent=0,revisions=0),
 dict(id='unknown_tool',transport='memory',profile='unknown_tool',state='failed',code='plan_invalid_proposal_schema',sent=0,revisions=0),
 dict(id='status_forgery',transport='memory',profile='status_forgery',state='failed',code='plan_invalid_proposal_schema',sent=0,revisions=0),
 dict(id='unplanned',transport='memory',profile='unplanned',state='failed',code='plan_action_not_in_plan',sent=0,revisions=1),
 dict(id='dependency',transport='memory',profile='skip_dependency',state='failed',code='plan_step_not_ready',sent=0,revisions=1),
 dict(id='premature',transport='memory',profile='premature',state='failed',code='verification_incomplete',sent=0,revisions=1),
 dict(id='stale_parent',transport='memory',ambiguous=True,profile='stale_parent',state='failed',code='plan_stale_plan_parent',sent=2,revisions=1),
 dict(id='erase_history',transport='memory',ambiguous=True,profile='erase_history',state='failed',code='plan_rewrote_executed_step',sent=2,revisions=1),
 dict(id='no_reason',transport='memory',ambiguous=True,profile='no_reason',state='failed',code='plan_unjustified_replan',sent=2,revisions=1),
 dict(id='unknown_evidence',transport='memory',ambiguous=True,profile='unknown_evidence',state='failed',code='plan_unknown_plan_evidence',sent=2,revisions=1),
 dict(id='repeated_query',transport='memory',ambiguous=True,profile='repeated_query',state='failed',code='plan_duplicate_plan_action',sent=2,revisions=1),
 dict(id='claim_root',transport='memory',profile='claim_root',state='failed',code='plan_unverified_conclusion',sent=3,revisions=1),
 dict(id='ignore_discrepancy',transport='memory',health_status='ok',profile='ignore_discrepancy',state='failed',code='unresolved_discrepancy',sent=3,revisions=1),
 dict(id='revision_budget',transport='memory',ambiguous=True,max_revisions=1,state='stopped',code='plan_revision_budget',sent=2,revisions=1),
 dict(id='tool_budget',transport='memory',ambiguous=True,max_tools=2,state='stopped',code='tool_budget',sent=2,revisions=2),
 dict(id='thrash',transport='memory',ambiguous=True,profile='thrash',state='failed',code='plan_reused_revision_trigger',sent=3,revisions=2),
]

def setup(path,case):
 w=World48(path)
 body=json.dumps(dict(symptom=case.get('symptom','gateway timeout 504 upstream trace_id'),reported_health='degraded',report_at=case.get('report_at',TIME)),ensure_ascii=False,sort_keys=True)
 try:w.core.store.db.execute('UPDATE tickets SET body=? WHERE id=1',(body,));seed=[dict(row) for row in w.core.store.db.execute('SELECT * FROM tickets ORDER BY id')]
 finally:w.close()
 question='mã XQ7 ở cổng khách hàng' if case.get('ambiguous') else 'gateway timeout 504 upstream trace_id'
 return task(question=question),digest(seed)

def execute(out,case):
 p=Path(out)/(case['id']+'.sqlite');t,seed=setup(p,case);health=case.get('health_status','degraded')
 with Link(p,transport=case['transport'],health_status=health,fault=case.get('fault')) as link:
  port=PlanningPort(case.get('profile','adaptive'),case.get('max_revisions',3));agent=PlannedAgent(MCPWorld(p,link,health_status=health),port,max_tools=case.get('max_tools',5))
  try:r=agent.run('demo-agent-a',t);manifest=agent.manifest;audit=link.audit();snapshot=link.snapshot()
  finally:agent.close()
 snapshot['closed']=link.closed;r['mcp']=snapshot
 actual=(r['state'],r['code'],snapshot['tools_calls_sent'],len(r['plans']))
 return dict(id=case['id'],task=t,seed_hash=seed,transport=case['transport'],correct=actual==(case['state'],case['code'],case['sent'],case['revisions']),expected=dict(state=case['state'],code=case['code'],sent=case['sent'],revisions=case['revisions']),result=r,audit=audit,manifest=manifest)

def baseline(out,case):
 p=Path(out)/('baseline-'+case['id']+'.sqlite');t,seed=setup(p,case)
 with Link(p,health_status=case.get('health_status','degraded')) as link:
  a,s,c=previous.mock();agent=MCPAgent(MCPWorld(p,link,health_status=case.get('health_status','degraded')),a,clock=c)
  try:r=agent.run('demo-agent-a',t)
  finally:agent.close()
 return dict(state=r['state'],code=r['code'],turns=r['turns'],calls=r['mcp']['tools_calls_sent'])

def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);rows=[execute(out,c) for c in CASES];pairs=[]
 for name in ['straight_memory','recover_memory','discrepancy']:
  case=next(c for c in CASES if c['id']==name);base=baseline(out,case);new=next(r['result'] for r in rows if r['id']==name)
  pairs.append(dict(id=name,baseline=base,planned=dict(state=new['state'],code=new['code'],turns=new['turns'],calls=new['mcp']['tools_calls_sent']),scope='predeclared scripted-policy comparison, not LLM benchmark'))
 manifest=dict(base49=source_hash(),implementation50=implementation_hash(),experiment_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),case_snapshot=digest(CASES),default_release=rows[0]['result']['release'])
 result=dict(cases=len(rows),correct=sum(x['correct'] for x in rows),rows=rows,comparison=pairs,manifest=manifest,experiment_id=digest(manifest)[:16],planning_turns=sum(x['result']['planning_turns'] for x in rows),mcp_calls=sum(x['result']['mcp']['tools_calls_sent'] for x in rows),accepted_plans=sum(len(x['result']['plans']) for x in rows),provider_http_attempts=0,live_llm_calls=0)
 (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result
