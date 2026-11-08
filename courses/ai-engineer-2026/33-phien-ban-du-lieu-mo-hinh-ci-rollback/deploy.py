"""Transactional local control plane; no real deployment or remote actions."""
from dataclasses import dataclass
from ci import recheck
from build import load
from model import score


def activate(registry,release,expected_generation,reason='promote',fault=None):
    if type(expected_generation) is not int or expected_generation<0:raise ValueError('generation')
    if reason not in {'promote','rollback'}:raise ValueError('reason')
    db=registry.db;db.execute('BEGIN IMMEDIATE')
    try:
        before=registry.state()
        if before['generation']!=expected_generation:raise ValueError('generation_conflict')
        if before['release']==release:raise ValueError('already_active')
        recheck(registry,release)
        warm=load(registry,release)
        score(warm['model'],warm['preprocessing'],[.5,0.0])
        if fault=='before_switch':raise RuntimeError('injected_before_switch')
        db.execute("UPDATE deployment SET generation=?,release=? WHERE slot='production'",(expected_generation+1,release))
        db.execute('INSERT INTO events(old_release,new_release,generation,reason) VALUES (?,?,?,?)',
                   (before['release'],release,expected_generation+1,reason))
        if fault=='after_switch':raise RuntimeError('injected_after_switch')
        db.execute('COMMIT')
    except BaseException:
        if db.in_transaction:db.execute('ROLLBACK')
        raise
    return registry.state()


@dataclass(frozen=True)
class Handle:
    release: str
    generation: int
    weights: tuple
    bias: float
    threshold: float

    def predict(self,x):
        model=dict(weights=self.weights,bias=self.bias)
        probability=score(model,{'feature_order':['latency_ratio','error_ratio'],'scale':[1.0,1.0]},x)
        return dict(release=self.release,generation=self.generation,score=probability,priority=int(probability>=self.threshold))


def pin(registry):
    state=registry.state()
    if state['release'] is None:raise ValueError('not_ready')
    values=load(registry,state['release'])
    from model import validate_model
    validate_model(values['model'],values['preprocessing'],values['contract'])
    m=values['model']
    return Handle(state['release'],state['generation'],tuple(m['weights']),m['bias'],m['threshold'])
