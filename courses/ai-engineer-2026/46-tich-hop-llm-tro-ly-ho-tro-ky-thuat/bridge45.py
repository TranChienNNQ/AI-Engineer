"""Reuse lesson 45 without editing/copying its ticket gateway or authorization policy."""
import hashlib,importlib,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'45-thiet-ke-du-an-ai-end-to-end'
NAMES=['corpus','retrieval','store','pipeline','api']
for name in NAMES:
    if not (BASE/(name+'.py')).is_file():raise RuntimeError('keep lesson 45 beside lesson 46')
    if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BASE/(name+'.py'):raise RuntimeError('module collision; run lesson 46 in a fresh Python process')
sys.path.insert(0,str(BASE))
try:
    corpus=importlib.import_module('corpus');retrieval=importlib.import_module('retrieval');store=importlib.import_module('store');pipeline=importlib.import_module('pipeline');api=importlib.import_module('api')
finally:sys.path.pop(0)
Engine45=pipeline.Engine;Fault=store.Fault;request=pipeline.request;digest=corpus.digest;documents=corpus.documents;TOKENS=corpus.TOKENS;retrieve=retrieval.retrieve;validate=retrieval.validate;create_app=api.create_app;RunIn=api.RunIn;RunOut=api.RunOut


def source_hash():return digest({n:hashlib.sha256((BASE/(n+'.py')).read_bytes()).hexdigest() for n in NAMES})
