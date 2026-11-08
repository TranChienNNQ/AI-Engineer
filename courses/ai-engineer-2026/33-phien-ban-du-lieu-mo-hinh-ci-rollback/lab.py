"""Local versioned release / CI / rollback lab. No cloud account, Git writes or external deployment."""
import argparse
import json
import platform
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from artifacts import Registry
from build import seed
from ci import gate
from demos import ci_demo,rollback_demo,hand_demo
from fixtures import evaluate


def report(path):
    with Registry(path) as r:
        seed(r);ci=ci_demo(r);rollback=rollback_demo(r)
        manifests={name:r.db.execute('SELECT manifest FROM releases WHERE id=?',(name,)).fetchone()[0]
                   for name in ['baseline','candidate','slice-regression']}
    return dict(metadata=dict(python=platform.python_version(),sqlite=sqlite3.sqlite_version,
                              transport='local SQLite and Python; no CI provider/container/cluster',
                              model='fixed weights, no training',source='actual chapter Python files and requirements snapshot'),
                manifests=manifests,ci=ci,rollback=rollback,hand=hand_demo(),evaluation=evaluate())


def run(mode,path,candidate):
    if mode=='all':return report(path)
    if mode=='check':return evaluate()
    if mode=='hand':return hand_demo()
    with Registry(path) as r:
        names=seed(r)
        if mode=='build':return dict(releases={n:r.manifest(n) for n in names})
        if mode=='ci':return gate(r,candidate)
        return rollback_demo(r)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['build','ci','rollback','check','hand','all'],default='all')
    p.add_argument('--candidate',choices=['baseline','candidate','slice-regression'],default='candidate')
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.out:
        args.out.mkdir(parents=True,exist_ok=False)
        result=run(args.mode,args.out/'registry.db',args.candidate)
    else:
        with TemporaryDirectory(prefix='lesson33-') as tmp:result=run(args.mode,Path(tmp)/'registry.db',args.candidate)
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False);print(text)
    if args.out:(args.out/'report.json').write_text(text+'\n')
    if args.mode=='ci' and not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
