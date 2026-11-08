"""Teaching bridge: preserve sibling lessons 45 and 46, use a fresh process."""
import hashlib,importlib,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'46-tich-hop-llm-tro-ly-ho-tro-ky-thuat'
NAMES=['bridge45','contracts','llm_adapter','fixtures','support']
for name in NAMES:
    target=BASE/(name+'.py')
    if not target.is_file():raise RuntimeError('keep lessons 45 and 46 beside lesson 47')
    if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=target:raise RuntimeError('module collision; use a fresh Python process')
sys.path.insert(0,str(BASE))
try:
    b=importlib.import_module('bridge45');contracts=importlib.import_module('contracts');adapter=importlib.import_module('llm_adapter');fixtures=importlib.import_module('fixtures');support=importlib.import_module('support')
finally:sys.path.pop(0)
SupportEngine=support.SupportEngine;ResponsesAdapter=adapter.ResponsesAdapter
request=b.request;digest=b.digest;documents=b.documents;TOKENS=b.TOKENS;retrieve=b.retrieve
Clock=fixtures.Clock;envelope=fixtures.envelope;reply_for=fixtures.reply_for

def source_hash():return digest(dict(base45=b.source_hash(),lesson46={n:hashlib.sha256((BASE/(n+'.py')).read_bytes()).hexdigest() for n in NAMES}))
