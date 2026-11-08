import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(name,'1')
import argparse
import json
import platform
from pathlib import Path
import tempfile
import numpy as np
import torch
import PIL
from arithmetic import hand
from data import export_examples
from training import train
from checks import run as check


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','hand','train','check','media'],default='all');p.add_argument('--out',type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    if a.mode in ['all','train','media']:
        if a.out is None:p.error('--out required; use a fresh directory')
        if a.out.exists():p.error('output already exists')
    if a.mode=='hand':r=hand()
    elif a.mode=='media':r=export_examples(a.out)
    elif a.mode=='train':r=train(a.out)
    elif a.mode=='check':
        with tempfile.TemporaryDirectory() as temp:r=check(Path(temp))
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        r=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,pillow=PIL.__version__,platform=platform.machine(),device='cpu',threads=1),
            hand=hand(),experiment=train(a.out/'training'),media=export_examples(a.out/'media'))
        r['checks']=check(a.out/'checks');(a.out/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
