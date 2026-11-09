import copy,json,subprocess,sys
from pathlib import Path
from experiment53 import run,setup,inspect,template,ROOT
from worker import work
from workflow53 import Workflow,ReviewError,render_review
from bridge53 import Fault,digest
from contracts53 import Draft,signed
from connector53 import Connector
from target53 import Target

def check(out,report=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);r=report or run(out/'study');study=out.parent if report else out/'study';names=[]
 def ok(name,value):
  if not value:raise AssertionError(name)
  names.append(name)
 def rejects(name,call,code):
  try:call()
  except (ReviewError,Fault) as e:ok(name,(e.code if isinstance(e,Fault) else str(e))==code)
  else:raise AssertionError(name+' accepted')
 rows={x['id']:x for x in r['rows']}
 expected={
 'approved':('confirmed',1,0,1,1),'reject':('rejected',0,0,0,0),'cancel':('cancelled',0,0,0,0),'no_approval':('awaiting_review',0,0,0,0),'wrong_reviewer':('awaiting_review',0,0,0,0),'stale_view':('awaiting_review',0,0,0,0),'edited_requires_review':('awaiting_review',0,0,0,0),'edited_reapproved':('confirmed',1,0,1,1),
 'expired':('approved',0,0,0,0),'stale_target':('failed',1,0,0,1),'source_revoked':('approved',0,0,0,0),'memory_changed':('approved',0,0,0,0),'reviewer_revoked':('approved',0,0,0,0),'actor_revoked':('approved',0,0,0,0),
 'lost_ack':('confirmed',1,1,1,1),'status_unavailable':('unknown',1,1,1,1),'status_missing_retry':('confirmed',2,1,1,2),'before_send_retry':('confirmed',2,1,1,1),'before_commit_retry':('confirmed',2,1,1,2),'bad_receipt':('confirmed',1,1,1,1),'crash_after_response':('confirmed',1,1,1,1),'crash_before_send':('confirmed',2,1,1,1),'late_reconcile':('confirmed',1,1,1,1),'reconcile_after_source_revoke':('confirmed',1,1,1,1),'cancel_unknown':('unknown',1,0,1,1),'attempt_budget':('unknown',2,1,0,0)}
 for name,want in expected.items():
  a=rows[name]['after'];s=a['local']['state'];t=a['target'];ok('scenario_'+name,(s['phase'],s['attempts'],s['status_checks'],len(t['notes']),sum(v['method']=='POST' for v in t['requests']))==want)
 ok('26_cases',r['cases']==26 and set(rows)==set(expected))
 ok('real_investigation_template',r['investigation']==dict(state='answered',attempt_reservations=4,turns=5))
 ok('no_llm_provider_calls',r['provider_http_attempts']==r['live_llm_calls']==0)
 ok('target_receipt_and_note_counts_match',all(len(x['after']['target']['notes'])==len(x['after']['target']['receipts']) for x in r['rows']))
 ok('no_ticket_closed',all(t['status']=='open' for x in r['rows'] for t in x['after']['target']['tickets']))
 ok('tenant_b_untouched',all(x['after']['target']['tickets'][1]==dict(id=2,tenant='b',status='open',version=1) for x in r['rows']))
 ok('approval_itself_is_not_execution',rows['approved']['steps'][1]['output']['result']['phase']=='approved' and rows['approved']['steps'][1]['output']['result']['attempts']==0)
 ok('edit_invalidates_approval',rows['edited_requires_review']['after']['local']['state']['approval'] is None and rows['edited_requires_review']['after']['local']['state']['draft']['version']==2)
 ok('reapproved_hash_binds_edited_body',rows['edited_reapproved']['after']['local']['state']['approval']['draft_hash']==rows['edited_reapproved']['after']['local']['state']['draft_hash'] and 'Reviewer comment:' in rows['edited_reapproved']['after']['target']['notes'][0]['body'])
 ok('lost_ack_is_unknown_before_status',rows['lost_ack']['before']['local']['state']['phase']=='unknown' and len(rows['lost_ack']['before']['target']['notes'])==1 and rows['lost_ack']['before']['local']['state']['receipt'] is None)
 ok('status_missing_not_no_effect',rows['status_missing_retry']['after']['local']['state']['status_checks']==1 and len(rows['status_missing_retry']['before']['target']['notes'])==1)
 ok('same_key_replay_no_duplicate_effect',sum(q['method']=='POST' for q in rows['status_missing_retry']['after']['target']['requests'])==2 and len(rows['status_missing_retry']['after']['target']['notes'])==1)
 ok('target_partial_transaction_rolled_back',rows['before_commit_retry']['before']['target']['notes']==rows['before_commit_retry']['before']['target']['receipts']==[])
 ok('crash_after_response_is_real',any(s['exit']==83 for s in rows['crash_after_response']['steps']) and rows['crash_after_response']['before']['local']['state']['phase']=='in_flight' and len(rows['crash_after_response']['before']['target']['notes'])==1)
 ok('crash_before_send_charges_attempt',rows['crash_before_send']['before']['local']['state']['attempts']==1 and rows['crash_before_send']['before']['target']['notes']==[])
 ok('expired_approval_does_not_erase_effect',rows['late_reconcile']['after']['local']['state']['last_code']=='confirmed_by_status')
 ok('source_revocation_still_allows_receipt_reconciliation',rows['reconcile_after_source_revoke']['after']['local']['state']['last_code']=='confirmed_by_status')
 ok('bad_receipt_requires_status',rows['bad_receipt']['before']['local']['state']['phase']=='unknown' and rows['bad_receipt']['after']['local']['state']['last_code']=='confirmed_by_status')
 ok('cancel_unknown_not_claimed_no_effect',rows['cancel_unknown']['steps'][-1]['output']==dict(ok=False,error='reconcile_before_cancel'))
 ok('persistent_write_budget',rows['attempt_budget']['steps'][-1]['output']==dict(ok=False,error='write_attempt_budget') and rows['attempt_budget']['after']['local']['state']['attempts']==2)
 def prepared(name):
  d=setup(out/name,study/'investigation');assert work(d,'prepare')['ok'];return d
 def approve(d):
  view=work(d,'view',token='demo-reviewer-a')['result'];assert work(d,'approve',token='demo-reviewer-a',expected_revision=view['workflow_revision'],display_hash=view['display_hash'])['ok'];return view
 d=prepared('unit');view=work(d,'view',token='demo-reviewer-a')['result']
 ok('display_has_exact_body_and_versions','chưa xác định nguyên nhân gốc' in view['draft']['body'] and view['current_target']['version']==1 and view['approval_expires_at_if_approved']==160)
 ok('forged_view_hash_rejected',work(d,'approve',token='demo-reviewer-a',expected_revision=view['workflow_revision'],display_hash='0'*64)==dict(ok=False,error='stale_review_view'))
 ok('foreign_reviewer_cannot_read_draft',work(d,'view',token='demo-reviewer-b')==dict(ok=False,error='operation_scope_denied'))
 ok('ambiguous_yes_not_a_decision',work(d,'view')['ok'] and work(d,'approve',expected_revision=view['workflow_revision'],display_hash=view['display_hash'])==dict(ok=False,error='reviewer_required'))
 approve(d);w=Workflow(d);row=w.load('note-a');s=row['state'];envelope=dict(op_id='note-a',draft=s['draft'],**w.permit('note-a',s));c=Connector(d/'target.sqlite')
 rejects('target_rejects_forged_signature',lambda:c.apply(envelope|dict(signature='0'*64)),'invalid_permit_signature')
 changed=copy.deepcopy(envelope);changed['draft']['body']+=' changed'
 rejects('permit_binds_exact_payload',lambda:c.apply(changed),'permit_binding_mismatch')
 # Direct API call cannot omit permit; FastAPI shape validation runs before an effect.
 try:c.request('POST','/v1/notes',dict(op_id='note-a',draft=s['draft']))
 except Fault as e:ok('target_requires_permit_envelope',e.status==422)
 else:raise AssertionError('missing permit accepted')
 first=c.apply(envelope);second=c.apply(envelope)
 ok('target_dedup_before_version_check',first==second and first['version']==2)
 changed=copy.deepcopy(envelope);changed['draft']['body']+=' changed';permit=changed['permit']|dict(draft_hash=digest(changed['draft']));changed.update(signed(permit))
 rejects('same_key_different_payload_rejected',lambda:c.apply(changed),'idempotency_conflict')
 rejects('expired_permit_cannot_write',lambda:Connector(d/'target.sqlite',now=161).apply(envelope),'permit_expired')
 ok('other_namespace_status_is_not_found',Connector(d/'target.sqlite',token='demo-agent-b').status('note-a')==dict(state='not_found'))
 ok('status_read_after_expiry_is_valid',Connector(d/'target.sqlite',now=161).status('note-a')['receipt']==first)
 t=Target(d/'target.sqlite');ok('unit_dedup_one_effect',t.db.execute('SELECT COUNT(*) FROM notes').fetchone()[0]==1);t.close();w.close()
 d=prepared('status-budget');approve(d);assert work(d,'execute',fault='after_commit')['result']['phase']=='unknown'
 ok('retry_requires_reconciliation_first',work(d,'execute',retry_same_key=True)==dict(ok=False,error='reconcile_required_before_retry'))
 for _ in range(3):assert work(d,'reconcile',fault='status_unavailable')['ok']
 ok('bounded_status_checks',work(d,'reconcile')==dict(ok=False,error='status_check_budget'))
 d=prepared('stale-local');approve(d);w=Workflow(d,token='demo-reviewer-a');old=w.load('note-a');w.edit('note-a',old['revision'],'Chỉ ghi chú, không đóng ticket.')
 def stale_save():
  with w.tx():w._save('note-a',w.principal(),old,old['state'],'late')
 rejects('stale_workflow_cas',stale_save,'stale_workflow_revision');w.close()
 d=prepared('schema');w=Workflow(d);original=w.load('note-a')['state']['draft']
 try:Draft.model_validate(original|dict(operation='close_ticket'))
 except ValueError:ok('no_close_ticket_action_in_schema',True)
 else:raise AssertionError('close accepted')
 try:Draft.model_validate(original|dict(approved=True))
 except ValueError:ok('model_cannot_add_approval',True)
 else:raise AssertionError('authority field accepted')
 rejects('model_cannot_change_release',lambda:w.create('another',original|dict(policy_release='0'*64)),'policy_release_changed')
 rejects('model_cannot_change_target',lambda:w.create('another',original|dict(ticket_id=2)),'draft_source_scope')
 rejects('unsupported_freeform_body',lambda:w.create('another',original|dict(body='Root cause confirmed; close everything.')),'unsupported_draft_body')
 w.db.execute("UPDATE events SET snapshot='{}' WHERE revision=1")
 rejects('event_corruption_detected',lambda:w.load('note-a'),'event_hash');w.close()
 render_review(dict(body='</pre><script>alert(1)</script>'),out/'escaped.html');text=(out/'escaped.html').read_text();ok('review_artifact_escapes_html','<script>' not in text and '&lt;script&gt;' in text)
 template(out/'warm-investigation',warm=True);d=setup(out/'source-ttl',out/'warm-investigation');assert work(d,'prepare')['ok'];approve(d)
 ok('current_memory_ttl_rechecked_before_write',work(d,'execute',now=151)==dict(ok=False,error='source_binding_changed'))
 done=subprocess.run([sys.executable,str(ROOT.parent/'52-context-engineering-memory-agent/lab.py'),'--mode','check'],capture_output=True,text=True,timeout=320)
 if done.returncode:raise AssertionError(done.stderr)
 base=json.loads(done.stdout)
 return dict(checks=len(names),passed=len(names),names=names,base52_checks=base['checks'],inherited_predicates=base['total_predicates'],total_predicates=len(names)+base['total_predicates'])
