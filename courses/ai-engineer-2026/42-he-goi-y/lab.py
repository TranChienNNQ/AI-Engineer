import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(name,'1')
import argparse
import json
import platform
from pathlib import Path
import tempfile
import numpy as np
import torch
from arithmetic import hand
from data import dataset,train_events,fingerprint
from training import train
from checks import run as check


def data_summary():
    d=dataset();return dict(catalog=len(d['catalog']),train_events=len(train_events(d)),validation_requests=len(d['validation']),test_requests=len(d['test']),
        test_cohorts={name:sum(r['cohort']==name for r in d['test']) for name in ['warm','new_item','new_user']},data_hash=fingerprint(d))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','train','check'],default='all');p.add_argument('--out',type=Path);a=p.parse_args();torch.set_num_threads(1)
    if a.mode in ['all','train']:
        if a.out is None:p.error('--out required; choose a fresh directory')
        if a.out.exists():p.error('output already exists')
    if a.mode=='hand':r=hand()
    elif a.mode=='data':r=data_summary()
    elif a.mode=='train':r=train(a.out)
    elif a.mode=='check':
        with tempfile.TemporaryDirectory() as t:r=check(Path(t))
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        r=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,platform=platform.machine(),device='cpu',threads=1),
            hand=hand(),data=data_summary(),experiment=train(a.out/'training'),checks=check(a.out/'checks'))
        (a.out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
