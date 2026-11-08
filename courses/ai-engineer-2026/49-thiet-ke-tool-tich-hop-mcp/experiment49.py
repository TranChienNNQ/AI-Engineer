import json,hashlib
from pathlib import Path
from bridge49 import Agent48,World48,mock,task,digest,source_hash
from gateway import MCPWorld,MCPAgent,implementation_hash
from mcp_link import Link

CASES=[
 dict(id='memory_triage',transport='memory',state='answered',code='ok',sent=3),
 dict(id='stdio_triage',transport='stdio',state='answered',code='ok',sent=3),
 dict(id='memory_retry',transport='memory',goal='retry',state='answered',code='ok',sent=2),
 dict(id='stdio_retry',transport='stdio',goal='retry',state='answered',code='ok',sent=2),
 dict(id='memory_unknown',transport='memory',health_status='unknown',state='handoff',code='need_info',sent=3),
 dict(id='stdio_unknown',transport='stdio',health_status='unknown',state='handoff',code='need_info',sent=3),
 dict(id='tool_error',transport='memory',fault='tool_error',state='handoff',code='tool_unavailable',sent=3),
 dict(id='unavailable',transport='memory',fault='unavailable',state='handoff',code='tool_unavailable',sent=3),
 dict(id='wrong_ticket',transport='memory',fault='wrong_ticket',state='failed',code='ticket_result_identity',sent=1),
 dict(id='bad_quote',transport='memory',fault='bad_quote',state='failed',code='source_result_identity',sent=2),
 dict(id='bad_health',transport='memory',fault='bad_health',state='failed',code='health_result_identity',sent=3),
 dict(id='timeout',transport='memory',fault='slow_health',timeout=.05,state='failed',code='mcp_timeout',sent=3),
 dict(id='role_denied',transport='memory',actor='demo-reader-a',state='blocked',code='health_scope_denied',sent=2),
 dict(id='target_denied',transport='memory',profile='cross_target',state='blocked',code='target_scope_denied',sent=0),
]

def execute(out,case):
 p=Path(out)/(case['id']+'.sqlite');kwargs=dict(goal=case.get('goal','triage'),actor=case.get('actor','demo-agent-a'),health_status=case.get('health_status','degraded'))
 with Link(p,transport=case['transport'],fault=case.get('fault'),timeout=case.get('timeout',.5),**kwargs) as link:
  a,s,c=mock(case.get('profile','adaptive'));world=MCPWorld(p,link,health_status=kwargs['health_status']);agent=MCPAgent(world,a,clock=c)
  try:
   result=agent.run(kwargs['actor'],task(goal=kwargs['goal']));manifest=agent.manifest;audit=link.audit();snapshot=link.snapshot()
  finally:agent.close()
 snapshot['closed']=link.closed;result['mcp']=snapshot
 return dict(id=case['id'],transport=case['transport'],correct=(result['state'],result['code'],snapshot['tools_calls_sent'])==(case['state'],case['code'],case['sent']),expected=dict(state=case['state'],code=case['code'],sent=case['sent']),result=result,audit=audit,manifest=manifest)

def direct(out,goal):
 a,s,c=mock();agent=Agent48(World48(Path(out)/('direct-'+goal+'.sqlite')),a,clock=c)
 try:return agent.run('demo-agent-a',task(goal=goal))
 finally:agent.close()

def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);rows=[execute(out,c) for c in CASES];index={x['id']:x for x in rows};pairs=[]
 for goal in ['triage','retry']:
  local=direct(out,goal);remote=index['stdio_'+goal]['result']
  pairs.append(dict(goal=goal,same_answer=local['answer']==remote['answer'],same_citations=local['citations']==remote['citations'],same_path=[o['tool'] for o in local['observations']]==[o['tool'] for o in remote['observations']],model_calls_direct=local['model_calls'],model_calls_mcp=remote['model_calls'],reads_direct=local['actual_reads'],mcp_calls=remote['mcp']['tools_calls_sent']))
 m=dict(experiment_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),base48=source_hash(),implementation49=implementation_hash(),case_snapshot=digest(CASES),catalog_hash=rows[0]['result']['mcp']['catalog_hash'],model_fixture='lesson48 AdaptiveScript; not LLM',default_release=rows[0]['result']['release'])
 report=dict(cases=len(rows),correct=sum(x['correct'] for x in rows),rows=rows,comparison=pairs,manifest=m,experiment_id=digest(m)[:16],model_calls=sum(x['result']['model_calls'] for x in rows),mcp_calls_sent=sum(x['result']['mcp']['tools_calls_sent'] for x in rows),server_attempted=sum(x['audit']['stats']['attempted'] for x in rows),server_admitted_reads=sum(x['audit']['admitted_reads'] for x in rows),live_llm_calls=0,scope='real MCP SDK memory and stdio subprocess transports; scripted Responses model; no TCP/OAuth/provider/production benchmark')
 (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return report
