"""Regression checks for actual local artifacts, CI and transactional switch."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path
from tempfile import TemporaryDirectory
from artifacts import Registry,canonical,digest
from build import build,seed,load
from ci import gate,approve,POLICY
from deploy import activate,pin
from demos import rollback_demo


def evaluate():
    rows=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        rows.append(dict(name=name,passed=True))
    def rejects(fn,error=ValueError):
        try:fn()
        except error:return True
        return False
    with TemporaryDirectory(prefix='lesson33-check-') as tmp:
        path=Path(tmp)/'registry.db'
        with Registry(path) as r:
            seed(r)
            check('canonical_key_order',canonical({'b':2,'a':1})==canonical({'a':1,'b':2}))
            check('different_threshold_different_digest',digest({'threshold':.5})!=digest({'threshold':.6}))
            check('nonfinite_not_serialized',rejects(lambda:canonical(float('nan'))))
            m=r.manifest('baseline');before=m['objects']
            after=r.manifest('candidate')['objects']
            check('threshold_changes_model_component',before['model']!=after['model'])
            check('identical_components_reused',all(before[k]==after[k] for k in before if k not in {'model','metrics'}))
            check('idempotent_registration',r.register('baseline',m)==digest(m))
            check('release_name_cannot_rebind',rejects(lambda:build(r,'baseline',.55)))
            check('unknown_object_rejected',rejects(lambda:r.get('0'*64)))
            check('digest_format_closed',rejects(lambda:r.get('../secret')))
            base=gate(r,'baseline');good=gate(r,'candidate');bad=gate(r,'slice-regression')
            check('baseline_ci_passes',base['passed'] and base['observed']['overall']['accuracy']==.9)
            check('candidate_ci_passes',good['passed'] and good['observed']['overall']['accuracy']==1)
            check('bad_overall_can_equal_baseline',bad['observed']['overall']['accuracy']==.9)
            check('bad_mobile_blocks_release',not bad['passed'] and bad['observed']['slices']['mobile']['accuracy']==.5)
            check('ci_failure_no_approval',rejects(lambda:approve(r,'slice-regression')))
            check('unknown_release_fail_closed',not gate(r,'missing')['passed'])
            mutations={
                'feature_order':lambda v:v['preprocessing'].update(feature_order=['error_ratio','latency_ratio']),
                'wrong_scale':lambda v:v['preprocessing'].update(scale=[1000.0,1.0]),
                'label_mapping':lambda v:v['contract'].update(label_mapping={'0':'urgent','1':'normal'}),
                'forged_metrics':lambda v:v['metrics']['overall'].update(accuracy=.123),
                'group_leakage':lambda v:v['validation']['rows'][0].update(group='train-g0'),
                'empty_validation':lambda v:v['validation'].update(rows=[]),
                'boolean_feature':lambda v:v['train']['rows'][0].update(x=[True,0.0]),
                'future_environment':lambda v:v['environment'].update(python='99.0.0'),
                'wrong_source':lambda v:v['source'].update({'model.py':'print("replaced")'}),
                'different_validation':lambda v:v['validation']['rows'][0].update(id='another-id'),
                'wrong_golden':lambda v:v['model'].update(bias=-1.0),
                'invalid_weight':lambda v:v['model'].update(weights=[4.0]),
            }
            for name,mutation in mutations.items():
                build(r,'bad-'+name.replace('_','-'),mutate=mutation)
                check('reject_'+name,not gate(r,'bad-'+name.replace('_','-'))['passed'])
            check('not_ready_without_active',rejects(lambda:pin(r)))
            check('unapproved_cannot_activate',rejects(lambda:activate(r,'candidate',0)))
            check('rejection_keeps_pointer',r.state()==dict(generation=0,release=None) and not r.history())
            approve(r,'baseline');approve(r,'candidate');activate(r,'baseline',0)
            for fault in ['before_switch','after_switch']:
                check('fault_'+fault,rejects(lambda:activate(r,'candidate',1,fault=fault),RuntimeError))
                check('fault_atomic_'+fault,r.state()==dict(generation=1,release='baseline') and len(r.history())==1)
            check('stale_generation_rejected',rejects(lambda:activate(r,'candidate',0)))
            handle=pin(r);activate(r,'candidate',1)
            check('inflight_pins_old_bundle',handle.predict([.6,0.0])['release']=='baseline' and handle.predict([.6,0.0])['priority']==0)
            check('new_request_uses_candidate',pin(r).predict([.6,0.0])['priority']==1)
            activate(r,'baseline',2,'rollback')
            check('rollback_increases_generation',r.state()==dict(generation=3,release='baseline'))
            check('rollback_restores_prediction',pin(r).predict([.6,0.0])['priority']==0)
            check('audit_matches_switches',[e['generation'] for e in r.history()]==[1,2,3])
            check('same_release_noop_rejected',rejects(lambda:activate(r,'baseline',3)))
            check('unknown_release_activation_rejected',rejects(lambda:activate(r,'missing',3)))
            # Deliberately bypass immutable API as a corruption fixture.
            key=r.manifest('candidate')['objects']['model'];original=r.db.execute('SELECT body FROM objects WHERE id=?',(key,)).fetchone()[0]
            r.db.execute('UPDATE objects SET body=? WHERE id=?',(b'{}',key))
            check('tamper_detected_by_digest',rejects(lambda:load(r,'candidate')))
            check('approved_tampered_release_blocked',rejects(lambda:activate(r,'candidate',3)))
            check('tamper_no_pointer_change',r.state()==dict(generation=3,release='baseline'))
            r.db.execute('UPDATE objects SET body=? WHERE id=?',(original,key))
            old=deepcopy(POLICY)
            try:
                POLICY['min_accuracy']=.95
                check('policy_change_stales_approval',rejects(lambda:activate(r,'candidate',3)))
            finally:POLICY.clear();POLICY.update(old)
        with Registry(path) as r:
            check('restart_recovers_pointer_and_audit',r.state()==dict(generation=3,release='baseline') and len(r.history())==3)
            check('restart_loads_prediction',pin(r).predict([.6,0.0])['priority']==0)
        race=Path(tmp)/'race.db'
        with Registry(race) as r:
            seed(r);build(r,'second-candidate',.55)
            for name in ['baseline','candidate','second-candidate']:approve(r,name)
            activate(r,'baseline',0)
        barrier=Barrier(2)
        def contender(name):
            with Registry(race) as r:
                barrier.wait(timeout=5)
                try:activate(r,name,1);return 'won'
                except ValueError as exc:return str(exc)
        with ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(contender,'candidate');b=pool.submit(contender,'second-candidate');result=[a.result(),b.result()]
        check('two_controllers_one_winner',sorted(result)==['generation_conflict','won'])
        with Registry(race) as r:check('race_one_committed_event',r.state()['generation']==2 and len(r.history())==2)
        with Registry(Path(tmp)/'demo.db') as r:
            seed(r);demo=rollback_demo(r)
            check('canary_detects_offline_gap',demo['canary']['candidate_accuracy']==.5 and demo['canary']['baseline_accuracy']==1)
            check('rollback_does_not_rewrite_old_handle',demo['request_pinned_before_rollback']['release']=='candidate')
    return dict(kind='Local release-pipeline regression, not model accuracy or production reliability',checks=len(rows),passed=len(rows),rows=rows)
