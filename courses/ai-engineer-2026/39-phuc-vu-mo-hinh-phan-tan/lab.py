import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(name,'1')
import argparse
import json
import platform
from pathlib import Path
import numpy as np
from arithmetic import hand
from shards import demo as shards
from fixtures import run as router
from checks import run as check


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','shards','router','check'],default='all');p.add_argument('--out',type=Path);a=p.parse_args()
    if a.mode=='all':
        if a.out is None:p.error('--out required; choose a fresh output directory')
        a.out.mkdir(parents=True,exist_ok=False)
    functions=dict(hand=hand,shards=shards,router=router,check=check)
    if a.mode=='all':
        r=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.machine(),transport='local multiprocessing spawn/Pipe'),
               hand=hand(),shards=shards(),router=router(),checks=check())
        r['verification']=dict(passed=r['checks']['passed']+r['router']['passed'],checks=r['checks']['checks']+r['router']['checks'])
        (a.out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    else:r=functions[a.mode]()
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
