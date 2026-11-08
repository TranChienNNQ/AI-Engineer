"""Deterministic local CI quality gate; the runner/DB are trusted in this exercise."""
from artifacts import digest
from build import load,source_snapshot,environment
from data import validate_data
from model import evaluate,validate_model,score

POLICY=dict(name='quality-policy-v1',min_accuracy=.9,min_slice_accuracy=.75,min_slice_n=2,allow_drop=0.0)


def gate(registry,release,baseline='baseline'):
    checks=[]
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    try:
        values=load(registry,release);base=load(registry,baseline)
        check('source_exact',values['source']==source_snapshot())
        check('environment_exact',values['environment']==environment())
        validate_data(values['train'],values['validation']);check('data_contract_and_split',True)
        validate_model(values['model'],values['preprocessing'],values['contract']);check('model_contract',True)
        validate_model(base['model'],base['preprocessing'],base['contract'])
        check('same_validation',registry.manifest(release)['objects']['validation']==registry.manifest(baseline)['objects']['validation'])
        actual=evaluate(values['model'],values['preprocessing'],values['validation']['rows'])
        reference=evaluate(base['model'],base['preprocessing'],values['validation']['rows'])
        check('stored_metrics_recomputed',values['metrics']==actual)
        check('golden_score',abs(score(values['model'],values['preprocessing'],[.5,0.0])-.5)<1e-12)
        check('minimum_accuracy',actual['overall']['accuracy']>=POLICY['min_accuracy'])
        check('no_overall_regression',actual['overall']['accuracy']+POLICY['allow_drop']>=reference['overall']['accuracy'])
        for segment,row in actual['slices'].items():
            check('slice_n_'+segment,row['n']>=POLICY['min_slice_n'])
            check('slice_accuracy_'+segment,row['accuracy'] is not None and row['accuracy']>=POLICY['min_slice_accuracy'])
        error=None
    except (ValueError,KeyError,TypeError,OverflowError) as exc:
        check('artifact_or_contract',False);actual=reference=None;error=type(exc).__name__
    return dict(release=release,baseline=baseline,manifest=registry.db.execute('SELECT manifest FROM releases WHERE id=?',(release,)).fetchone()[0] if registry.db.execute('SELECT 1 FROM releases WHERE id=?',(release,)).fetchone() else None,
                policy=digest(POLICY),passed=all(r['passed'] for r in checks),checks=checks,
                observed=actual,reference=reference,error=error)


def approve(registry,release,baseline='baseline'):
    report=gate(registry,release,baseline)
    if not report['passed']:raise ValueError('ci_rejected')
    key=registry.put(report)
    registry.db.execute('INSERT OR REPLACE INTO approvals VALUES (?,?,?,?)',(release,baseline,digest(POLICY),key))
    return report


def recheck(registry,release):
    row=registry.db.execute('SELECT baseline,policy,report FROM approvals WHERE release=?',(release,)).fetchone()
    if row is None:raise ValueError('approval_missing')
    baseline,policy,report_key=row
    old=registry.get(report_key);now=gate(registry,release,baseline)
    if policy!=digest(POLICY) or not now['passed'] or old!=now:raise ValueError('approval_stale_or_changed')
