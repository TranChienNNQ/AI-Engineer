import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(name,'1')
import argparse
import json
from pathlib import Path
import platform
import tempfile
import numpy as np
from arithmetic import hand
from data import tabular,temporal,fingerprint,TAB_SPEC,TIME_SPEC
from experiment import run
from checks import run as check


def data_summary():
    a,b=tabular(),temporal()
    return dict(tabular=dict(spec=TAB_SPEC,hash=fingerprint(a,['x','label']),validation_positives=int(a['label'][500:620].sum()),test_positives=int(a['label'][620:].sum())),temporal=dict(spec=TIME_SPEC,hash=fingerprint(b,['t','y','plan','plan_available_at','label']),test_positive_points=int(b['label'][336:].sum())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','experiment','check'],default='all');p.add_argument('--out',type=Path);a=p.parse_args()
    if a.mode in ['all','experiment']:
        if a.out is None:p.error('--out required; choose a fresh directory')
        if a.out.exists():p.error('output already exists')
    if a.mode=='hand':r=hand()
    elif a.mode=='data':r=data_summary()
    elif a.mode=='check':
        with tempfile.TemporaryDirectory() as directory:r=check(directory)
    elif a.mode=='experiment':r=run(a.out)
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        r=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.machine(),device='cpu'),hand=hand(),data=data_summary(),experiment=run(a.out/'experiment'),checks=check(a.out/'checks'))
        (a.out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
