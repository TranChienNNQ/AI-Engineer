"""Local context/authorization lab. No network, shell tools, credentials or real ticket service."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
from data import make_world
from security import Gateway,digest
from context import Config,ContextCache,build_context
from fixtures import evaluate_actions,evaluate_context,residual_case,close,read


def hand_demo():
    budget=1024-128-96-32-128
    return dict(context_budget=budget,grant_issued_at=100,ttl=30,expires_at=130,
                at_129_valid=129<130,at_130_valid=130<130,
                hypothetical_forbidden_proposal=.4,hypothetical_gate_allows_given_forbidden=.1,
                hypothetical_joint_bad_effect=.4*.1,
                illustrative_benign_success=5/6,illustrative_attack_success=2/12)


def permission_matrix(world):
    return [dict(subject=p.id,tenant=p.tenant,roles=list(p.roles),clearance=p.clearance,
                 readable=[r.id for r in world.resources.values() if world.can_read(p,r)],
                 closable_by_role=[r.id for r in world.resources.values() if world.can_close(p,r)])
            for p in world.principals.values()]


def report():
    w=make_world();cache=ContextCache()
    context=build_context(w,'session_alice','retry API timeout',cache=cache)
    warm=build_context(w,'session_alice','retry API timeout',cache=cache)
    return dict(kind='scripted adversarial proposals and deterministic authorization; no LLM attack benchmark',
                metadata=dict(python=platform.python_version(),jsonschema=version('jsonschema'),device='CPU'),
                dataset_hash=digest(dict(principals=[asdict(p) for p in w.principals.values()],
                                         resources=[asdict(r) for r in w.resources.values()],
                                         records=[asdict(r) for r in w.records.values()])),
                config=asdict(Config()),permissions=permission_matrix(w),context_example=context,
                warm_cache_hit=warm['cache_hit'],actions=evaluate_actions(),
                context_checks=evaluate_context(),residual=residual_case(),hand=hand_demo())


def grant_demo():
    w=make_world();gateway=Gateway(w);proposal=close()
    before=gateway.execute('session_ops_blue',proposal,now=1)
    grant=gateway.issue_after_review('session_ops_blue',proposal,now=10,ttl=30)
    approved=gateway.execute('session_ops_blue',proposal,now=11,grant=grant)
    replay=gateway.execute('session_ops_blue',proposal,now=12,grant=grant)
    return dict(before_grant=before,after_review=approved,replay=replay,audit=gateway.audit)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['hand','context','grant','eval','all'],default='hand')
    parser.add_argument('--session',default='session_alice',help='Synthetic fixture session, not real authentication')
    parser.add_argument('--question',default='retry API timeout');parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.out and args.out.exists():raise FileExistsError('Choose a new output directory')
    if args.mode=='hand':obj=hand_demo()
    elif args.mode=='context':obj=build_context(make_world(),args.session,args.question,cache=ContextCache())
    elif args.mode=='grant':obj=grant_demo()
    else:obj=report()
    print(json.dumps(obj,ensure_ascii=False,indent=2))
    if args.out:
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/'report.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))
        ctx=obj.get('context_example') if args.mode in {'all','eval'} else obj if args.mode=='context' else None
        if ctx:(args.out/'prompt.txt').write_text(ctx['prompt'])
        if args.mode=='all':(args.out/'grant-demo.json').write_text(json.dumps(grant_demo(),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
