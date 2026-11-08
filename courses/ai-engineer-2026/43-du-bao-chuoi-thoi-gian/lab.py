import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(name,'1')
import argparse
import json
import platform
from pathlib import Path
import tempfile
import numpy as np
import torch
from data import dataset,origins,fingerprint
from arithmetic import hand
from experiment import run
from checks import run as check


def data_summary():
    d=dataset();return dict(samples=420,start_date='2025-01-06',initial_end=224,validation_origins=origins(224,280),calibration_origins=origins(280,364),test_origins=origins(364,420),data_hash=fingerprint(d))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','data','experiment','check'],default='all');p.add_argument('--out',type=Path);a=p.parse_args();torch.set_num_threads(1)
    if a.mode in ['all','experiment']:
        if a.out is None:p.error('--out required; use a fresh output directory')
        if a.out.exists():p.error('output already exists')
    if a.mode=='hand':r=hand()
    elif a.mode=='data':r=data_summary()
    elif a.mode=='experiment':r=run(a.out)
    elif a.mode=='check':
        with tempfile.TemporaryDirectory() as t:r=check(Path(t))
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        r=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,platform=platform.machine(),device='cpu',threads=1),hand=hand(),data=data_summary(),experiment=run(a.out/'experiment'),checks=check(a.out/'checks'))
        (a.out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
