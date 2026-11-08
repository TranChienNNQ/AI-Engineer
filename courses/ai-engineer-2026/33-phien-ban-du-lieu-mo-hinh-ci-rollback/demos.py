"""Small repeatable release/rollback experiment; all actions stay in a temp registry."""
from artifacts import Registry,digest
from build import seed
from ci import gate,approve
from deploy import activate,pin


def ci_demo(registry):
    return {name:gate(registry,name) for name in ['baseline','candidate','slice-regression']}


def rollback_demo(registry):
    approve(registry,'baseline');approve(registry,'candidate')
    first=activate(registry,'baseline',0);old_handle=pin(registry)
    promoted=activate(registry,'candidate',1);new_handle=pin(registry)
    # Distinct synthetic post-release cohort; never used to choose the offline threshold.
    live=[dict(x=[.6,0.0],y=0) for _ in range(5)]+[dict(x=[.9,0.0],y=1) for _ in range(5)]
    new_accuracy=sum(new_handle.predict(r['x'])['priority']==r['y'] for r in live)/len(live)
    old_accuracy=sum(old_handle.predict(r['x'])['priority']==r['y'] for r in live)/len(live)
    rolled=activate(registry,'baseline',2,reason='rollback')
    return dict(initial=first,promoted=promoted,rolled_back=rolled,
                request_pinned_before_promote=old_handle.predict([.6,0.0]),
                request_pinned_before_rollback=new_handle.predict([.6,0.0]),
                request_after_rollback=pin(registry).predict([.6,0.0]),
                canary=dict(n=10,candidate_accuracy=new_accuracy,baseline_accuracy=old_accuracy,
                            decision='ROLLBACK_IN_LOCAL_DEMO',labels='synthetic, immediate; no statistical claim'),
                history=registry.history())


def hand_demo():
    return dict(combinations=2**6,pinned_bundles=2,
                baseline_correct=9,candidate_correct=10,n=10,
                baseline_accuracy=.9,candidate_accuracy=1.0,gain_percentage_points=10,
                slice_regression_overall=.9,mobile_correct=1,mobile_n=2,mobile_accuracy=.5,
                canary_expected_requests=10000*.05,canary_expected_errors=10000*.05*.02,
                estimated_exposure=120*15,exposure_note='hypothetical steady rate × detection-and-switch seconds, not a guarantee',
                lineage_digest=digest({'weights':[4.0,1.0],'threshold':.5}))


def prepare(path):
    registry=Registry(path);seed(registry);return registry
