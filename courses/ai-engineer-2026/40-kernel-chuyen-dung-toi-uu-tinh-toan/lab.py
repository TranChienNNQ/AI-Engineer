import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(name,'1')
import argparse
import json
import platform
from pathlib import Path
import tempfile
import numpy as np
from native import Native
from arithmetic import hand
from algorithms import demo
from checks import run as check
from benchmark import run as benchmark


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','algorithms','check','bench'],default='all');p.add_argument('--out',type=Path);a=p.parse_args()
    if a.mode=='all':
        if a.out is None:p.error('--out required; use a fresh directory')
        a.out.mkdir(parents=True,exist_ok=False)
    if a.mode=='hand':r=hand()
    elif a.mode=='algorithms':r=demo()
    else:
        with tempfile.TemporaryDirectory(prefix='lesson40-native-') as build:
            native=Native(build)
            if a.mode=='check':r=check(native)
            elif a.mode=='bench':r=benchmark(native)
            else:
                r=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.machine(),system=platform.system(),accelerator='CPU only'),
                    hand=hand(),algorithms=demo(),checks=check(native),benchmark=benchmark(native))
                (a.out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
