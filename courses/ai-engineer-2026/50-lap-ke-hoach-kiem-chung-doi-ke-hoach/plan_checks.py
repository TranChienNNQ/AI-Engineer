import copy,json,subprocess,sys
from pathlib import Path
from bridge50 import digest,task,Link,MCPWorld
from planning import PlanningPort,PlannedAgent
from plan_guard import PlanGuard,PlanError
from plan_contract import Plan,Action,Proposal
from policy50 import initial_plan,tool_action,revised
from verification import verify,postcondition
from experiment50 import run


def check(out,report=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);names=[]
 def ok(name,cond):
  if not cond:raise AssertionError(name)
  names.append(name)
 def rejects(name,fn,code=None):
  try:fn()
  except PlanError as e:
   if code and str(e)!=code:raise AssertionError((name,str(e)))
   names.append(name)
  except (ValueError,TypeError):
   if code:raise
   names.append(name)
  else:raise AssertionError(name)
 if report is None:report=run(out/'suite')
 index={r['id']:r['result'] for r in report['rows']}
 for row in report['rows']:ok('scenario_'+row['id'],row['correct'])
 ok('25_scenarios',report['cases']==report['correct']==25)
 ok('two_stdio_scenarios',sum(r['transport']=='stdio' for r in report['rows'])==2)
 ok('all_mcp_closed',all(r['result']['mcp']['closed'] for r in report['rows']))
 ok('no_agent_ticket_effects',all(r['audit']['effects']==dict(tickets=2,runs=0,proposals=0,receipts=0,audit=0) for r in report['rows']))
 ok('recover_empty_then_success',[postcondition(o) for o in index['recover_memory']['observations']]==['satisfied','empty','satisfied','satisfied'])
 ok('old_empty_step_preserved',index['recover_memory']['step_states']['S2']=='empty' and index['recover_memory']['step_states']['S4']=='satisfied')
 ok('discrepancy_is_not_root_cause',next(c['status'] for c in index['discrepancy']['verification'] if c['check']=='root_cause_established')=='unknown')
 ok('different_times_not_contradiction',next(c['status'] for c in index['different_time']['verification'] if c['check']=='report_consistency')=='not_comparable')
 ok('tool_error_remains_distinct',postcondition(index['tool_failure']['observations'][-1])=='tool_error')
 ok('unknown_health_not_satisfied',postcondition(index['unknown_health']['observations'][-1])=='unknown')
 ok('no_provider_usage_claim',report['provider_http_attempts']==report['live_llm_calls']==0 and all(u['usage_source']=='not_applicable_scripted_port' for row in report['rows'] for u in row['result']['usage']))
 ok('baseline_recovery_difference',report['comparison'][1]['baseline']['state']=='handoff' and report['comparison'][1]['planned']['state']=='answered')
 ok('baseline_discrepancy_policy_difference',report['comparison'][2]['baseline']['state']=='answered' and report['comparison'][2]['planned']['state']=='handoff')
 t=task();base=initial_plan(t)
 g=PlanGuard();entry=g.accept(base,[],t)
 ok('initial_dag_only_ticket_ready',g.states([])=={'S1':'ready','S2':'pending','S3':'pending'})
 bad=copy.deepcopy(base);bad['steps'][1]['depends_on']=['missing']
 rejects('unknown_dependency',lambda:PlanGuard().accept(bad,[],t),'unknown_dependency')
 bad=copy.deepcopy(base);bad['steps'][1]['id']='S1'
 rejects('duplicate_step_id',lambda:PlanGuard().accept(bad,[],t),'duplicate_step_id')
 bad=copy.deepcopy(base);bad['steps'][1]['postcondition']='health_known'
 rejects('postcondition_mismatch',lambda:PlanGuard().accept(bad,[],t),'postcondition_mismatch')
 bad=copy.deepcopy(base);bad['steps'][0]['args']['ticket_id']=True
 rejects('strict_plan_arguments',lambda:PlanGuard().accept(bad,[],t),'invalid_plan_arguments')
 bad=copy.deepcopy(base);bad['steps'][0]['args']['ticket_id']=2
 rejects('plan_target_scope',lambda:PlanGuard().accept(bad,[],t),'plan_scope_denied')
 bad=copy.deepcopy(base);bad['steps'][1]['depends_on']=[]
 rejects('ticket_ancestor_required',lambda:PlanGuard().accept(bad,[],t),'ticket_prerequisite_missing')
 rejects('retry_health_plan_scope',lambda:PlanGuard().accept(initial_plan(task()),[],task(goal='retry')),'plan_scope_denied')
 bad=copy.deepcopy(base);bad['revision']=2
 rejects('initial_revision_not_two',lambda:PlanGuard().accept(bad,[],t),'invalid_initial_plan')
 bad=copy.deepcopy(base);bad['parent_hash']='0'*64
 rejects('initial_parent_null',lambda:PlanGuard().accept(bad,[],t),'invalid_initial_plan')
 bad=copy.deepcopy(base);bad['based_on']=['O999']
 rejects('initial_fake_evidence',lambda:PlanGuard().accept(bad,[],t),'unknown_plan_evidence')
 bad=copy.deepcopy(base);bad['steps'][0]['status']='satisfied'
 rejects('model_cannot_mark_complete',lambda:PlanGuard().accept(bad,[],t),'invalid_plan_schema')
 rejects('missing_plan',lambda:PlanGuard().authorize(dict(plan=None,action=tool_action(base['steps'][0])),[],t),'missing_plan')
 ready=tool_action(base['steps'][0]);p=Proposal.model_validate(dict(plan=None,action=ready)).model_dump()
 ok('action_ready_authorized',g.authorize(p,[],t)['action']['step_id']=='S1')
 mismatch=copy.deepcopy(p);mismatch['action']['args']['ticket_id']=2
 rejects('action_args_bound_to_plan',lambda:g.authorize(mismatch,[],t),'action_not_in_plan')
 obs=copy.deepcopy(index['recover_memory']['observations'][:2])
 old=g.current();new=revised(old,entry['hash'],'missing_evidence',['O2'],'request_clarification')
 e=g.accept(new,obs,t)
 ok('intent_change_is_revision',len(g.history)==2 and e['plan']['intent']=='request_clarification')
 again=revised(new,e['hash'],'missing_evidence',['O2'],'investigate')
 rejects('same_trigger_not_progress',lambda:g.accept(again,obs,t),'reused_revision_trigger')
 d=PlanGuard();d.accept(base,[],t)
 noop=revised(base,d.history[-1]['hash'],'missing_evidence',['O2'])
 rejects('no_op_revision',lambda:d.accept(noop,obs,t),'no_plan_change')
 repeat_action=tool_action(base['steps'][0])
 rejects('already_satisfied_not_ready',lambda:d.authorize(dict(plan=None,action=repeat_action),obs,t),'step_not_ready')
 fake=copy.deepcopy(base);fake['budget']=999
 rejects('no_model_budget_field',lambda:PlanGuard().accept(fake,[],t),'invalid_plan_schema')
 rejects('invalid_revision_budget',lambda:PlanGuard(True))
 # History entries must bind the exact accepted plan; no rewrite of previous executed step.
 ok('plan_hashes_match',all(p['hash']==digest(p['plan']) for row in report['rows'] for p in row['result']['plans']))
 ok('versions_monotonic',all([p['plan']['revision'] for p in row['result']['plans']]==list(range(1,len(row['result']['plans'])+1)) for row in report['rows']))
 p=out/'revoke.sqlite'
 with Link(p) as link:
  world=MCPWorld(p,link);port=PlanningPort();agent=PlannedAgent(world,port);original=port.script.propose
  def revoke(view):
   if len(port.script.contexts)==1:world.core.identities['demo-agent-a']['active']=False
   return original(view)
  port.script.propose=revoke
  try:
   revoked=agent.run('demo-agent-a',task())
   ok('revoked_actor_stops_dispatch',revoked['code']=='unauthorized' and link.sent==1)
   ok('planning_export_redacted',revoked['planning_export_redacted'] and not revoked['plans'] and not revoked['planning_trace'] and not revoked['observations'])
  finally:agent.close()
 old=Path(__file__).resolve().parent.parent/'49-thiet-ke-tool-tich-hop-mcp'
 done=subprocess.run([sys.executable,'lab.py','--mode','check'],cwd=old,capture_output=True,text=True,timeout=150)
 if done.returncode:raise AssertionError(done.stderr)
 previous=json.loads(done.stdout)
 if previous['passed']!=64 or previous['total_predicates']!=286:raise AssertionError('inherited checks failed')
 return dict(checks=len(names),passed=len(names),names=names,base49_checks=64,base48_checks=69,base46_checks=78,base45_checks=75,total_predicates=len(names)+286)
