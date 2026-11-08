import copy,json,sqlite3,subprocess,sys
from pathlib import Path
from experiment51 import run,setup,launch,inspect,ROOT
from journal51 import Journal,JournalError
from contracts51 import State,Limits,operation_id,recovery_rule
from worker import work
from runtime51 import restore

def check(out,report=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);r=report or run(out/'study');names=[]
 def ok(name,value):
  if not value:raise AssertionError(name)
  names.append(name)
 def rejects(name,call,code):
  try:call()
  except JournalError as e:ok(name,str(e)==code)
  else:raise AssertionError(name+' accepted')
 expected={
 'uninterrupted':('answered','ok',5,4,4),'pause_resume':('answered','ok',5,4,4),'planning_reserved':('answered','ok',6,4,4),
 'intent_committed':('answered','ok',5,4,4),'attempt_reserved':('answered','ok',5,5,4),'after_response':('answered','ok',5,5,5),
 'observation_committed':('answered','ok',5,4,4),'terminal_transaction':('answered','ok',6,4,4),'terminal_committed':('answered','ok',5,4,4),
 'attempt_budget':('stopped','tool_attempt_budget',2,2,2),'per_op_budget':('stopped','operation_attempt_budget',2,2,2),
 'expired':('stopped','run_expired',2,2,2),'turn_budget':('stopped','turn_budget',1,0,0)}
 rows={x['id']:x for x in r['rows']}
 for name,want in expected.items():
  a=rows[name]['after'];s=a['state'];ok('scenario_'+name,(s['phase'],s['result']['code'],s['turns'],s['attempts'],len(a['acknowledgements']))==want)
 ok('13_declared_cases',set(rows)==set(expected) and r['cases']==13)
 ok('25_fresh_process_launches',r['process_launches']==25)
 ok('actual_abrupt_deaths',sum(x['first']['exit']==83 for x in r['rows'])==9 and rows['terminal_transaction']['first']['exit']==84)
 ok('all_recovery_processes_succeeded',all(x['second'] is None or x['second']['exit']==0 and x['second']['output']['ok'] and x['second']['output']['link_closed'] for x in r['rows']))
 ok('ticket_seed_unchanged_after_setup',all(x['after']['seed_unchanged'] for x in r['rows']))
 ok('no_agent_business_effects',all(x['after']['effects']==dict(tickets=2,runs=0,proposals=0,receipts=0,audit=0) for x in r['rows']))
 a=rows['after_response'];ack=a['after']['acknowledgements'];ops=[x['op_id'] for x in ack]
 ok('unknown_gap_has_external_ack_not_observation',len(a['before']['acknowledgements'])==2 and len(a['before']['state']['observations'])==1 and a['before']['state']['pending']['op_id']=='support-504:S2')
 ok('same_operation_two_physical_acknowledgements',ops.count('support-504:S2')==2 and [x['attempt'] for x in ack if x['op_id']=='support-504:S2']==[1,2])
 ok('one_committed_observation_per_operation',len({o['call_id'] for o in a['after']['state']['observations']})==4)
 ok('epochs_distinguish_new_session',{x['epoch'] for x in ack}=={1,2})
 ok('reservation_before_send_can_overcount',rows['attempt_reserved']['after']['state']['attempts']==5 and len(rows['attempt_reserved']['after']['acknowledgements'])==4)
 ok('committed_observation_not_reexecuted',len(rows['observation_committed']['after']['acknowledgements'])==4)
 ok('atomic_partial_terminal_rolled_back',rows['terminal_transaction']['before']['state']['phase']=='planning' and not any(c['kind']=='finished' for c in rows['terminal_transaction']['before']['checkpoints']))
 ok('terminal_redelivery_no_new_planning',rows['terminal_committed']['before']['state']==rows['terminal_committed']['after']['state'])
 ok('pause_releases_owner',rows['pause_resume']['before']['header']['owner'] is None and rows['pause_resume']['before']['state']['phase']=='paused')
 ok('lost_planner_reservation_charged',rows['planning_reserved']['after']['state']['turns']==6)
 ok('suite_counters',(r['planning_reservations'],r['attempt_reservations'],r['completed_acknowledgements'])==(54,44,43))
 ok('no_llm_sampling',r['provider_http_attempts']==r['live_llm_calls']==0)
 ok('plans_reconstructed_on_resume',all(len(rows[n]['after']['state']['plans'])==2 for n in ['pause_resume','after_response','terminal_transaction']))
 ok('root_cause_not_established',all(x['second'] is None or next(c['status'] for c in x['second']['output']['result']['verification'] if c['check']=='root_cause_established')=='unknown' for x in r['rows']))
 j=Journal(out/'ownership.sqlite');binding={'version':'test'};j.create('r','a',binding)
 rejects('duplicate_run_id',lambda:j.create('r','a',binding),'run_exists')
 rejects('wrong_actor_cannot_load',lambda:j.load('r','other'),'run_actor_mismatch')
 rejects('different_binding',lambda:j.claim('r','a',{},'w1'),'run_binding_changed')
 h1=j.claim('r','a',binding,'w1');s=j.load('r','a')['state']
 rejects('second_worker_without_takeover',lambda:j.claim('r','a',binding,'w2'),'run_busy')
 h2=j.claim('r','a',binding,'w2',takeover_epoch=h1['epoch'])
 rejects('old_epoch_cannot_save',lambda:j.save('r','a',h1,s,'old'),'stale_owner')
 h3=j.save('r','a',h2,s,'tick')
 rejects('stale_revision_cannot_save',lambda:j.save('r','a',h2,s,'old'),'stale_revision')
 s['turns']=1;h4=j.save('r','a',h3,s,'planning_reserved');bad=copy.deepcopy(s);bad['turns']=0
 rejects('budget_cannot_regress',lambda:j.save('r','a',h4,bad,'reset'),'counter_regression')
 j.db.execute("UPDATE checkpoints SET state='{}' WHERE run_id='r' AND revision=1")
 rejects('corruption_detected',lambda:j.load('r','a'),'checkpoint_hash');j.close()
 for name,patch,flags,want,pause in [
  ('changed_task',{'task':dict(rows['uninterrupted']['first']['output']['result']['observations'][0]['args'],service='gateway',goal='triage',question='different question')},{},'run_binding_changed',2),
  ('changed_limit',{'limits':Limits(attempts=5).model_dump()},{},'run_binding_changed',2),
  ('revoked_source',{},dict(revoke=True),'source_no_longer_allowed',3),
  ('inactive_actor',{},dict(deny=True),'unauthorized',2),
  ('changed_health',{},dict(health='ok'),'run_binding_changed',2)]:
  d=out/name;setup(d);launch(d,pause_after=pause);before=inspect(d)
  result=work(d,owner='worker2',config_patch=patch,**flags)
  ok(name+'_blocks_export',result==dict(ok=False,error=want))
  ok(name+'_no_new_tool_ack',inspect(d)['acknowledgements']==before['acknowledgements'])
 # Revalidate even an already committed answer before returning its text.
 d=out/'cached';setup(d);launch(d)
 ok('cached_answer_rechecks_source',work(d,owner='worker2',revoke=True)==dict(ok=False,error='source_no_longer_allowed'))
 cached=work(d,owner='worker3',now=161)
 ok('expiry_stops_execution_not_validated_cached_delivery',cached['ok'] and cached['result']['turns']==5)
 cfg=json.loads((d/'config.json').read_text());j=Journal(d/'state.sqlite');loaded=j.load('support-504','a:agent-a');binding=json.loads(loaded['header']['binding']);h=j.claim('support-504','a:agent-a',binding,'w-final');s=loaded['state'];s['result']['answer']='edited'
 rejects('terminal_is_immutable',lambda:j.save('support-504','a:agent-a',h,s,'edit'),'terminal_immutable');j.close()
 d=out/'history';setup(d);launch(d,pause_after=2);j=Journal(d/'state.sqlite');loaded=j.load('support-504','a:agent-a');h=j.claim('support-504','a:agent-a',json.loads(loaded['header']['binding']),'w');s=loaded['state'];s['observations'][0]['data']['body']='rewritten'
 rejects('committed_history_is_immutable',lambda:j.save('support-504','a:agent-a',h,s,'edit'),'history_rewrite');j.close()
 ok('operation_ids_stable',operation_id('run','S2')=='run:S2' and operation_id('run','S2')!=operation_id('run','S4'))
 ok('write_not_retry_allowlisted',recovery_rule('commit_ticket')=='reconcile_or_review')
 for name,bad in [('pending_shape',dict(phase='pending')),('schema_version',dict(version='99.0')),('extra_authority',dict(approved=True))]:
  try:State.model_validate(bad)
  except ValueError:ok(name,True)
  else:raise AssertionError(name)
 done=subprocess.run([sys.executable,str(ROOT.parent/'50-lap-ke-hoach-kiem-chung-doi-ke-hoach'/'lab.py'),'--mode','check'],capture_output=True,text=True,timeout=140)
 if done.returncode:raise AssertionError(done.stderr)
 base=json.loads(done.stdout)
 return dict(checks=len(names),passed=len(names),names=names,base50_checks=base['checks'],inherited_predicates=base['total_predicates'],total_predicates=len(names)+base['total_predicates'])
