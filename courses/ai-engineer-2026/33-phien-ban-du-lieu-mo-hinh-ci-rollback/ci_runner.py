"""Run local CI against an existing bundle; approve only after all checks pass."""
import argparse
import json
from pathlib import Path
from artifacts import Registry
from build import ROOT
from ci import gate,approve
from fixtures import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--release',choices=['baseline','candidate','slice-regression'],default='candidate')
    p.add_argument('--approve',action='store_true')
    args=p.parse_args()
    if not args.registry.is_file():raise FileNotFoundError('Build the registry first')
    for path in ROOT.glob('*.py'):compile(path.read_text(),str(path),'exec')
    regression=evaluate()
    with Registry(args.registry) as registry:
        result=gate(registry,args.release)
        if result['passed'] and args.approve:approve(registry,args.release)
    print(json.dumps(dict(compilation='passed',regression=regression,release_gate=result,
                          approved=bool(result['passed'] and args.approve)),ensure_ascii=False,indent=2))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
