"""Reuse the single-agent runtime of lesson48, not a second independent agent loop."""
import importlib,sys,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'48-tu-tro-ly-rag-den-ai-agent'
NAMES=['bridge','agent_contract','provider','tools','runtime','fixtures']
for name in NAMES:
 if not (BASE/(name+'.py')).is_file():raise RuntimeError('keep lessons45,46,48 beside lesson49')
 if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BASE/(name+'.py'):raise RuntimeError('module collision; use a fresh Python process')
sys.path.insert(0,str(BASE))
try:
 b=importlib.import_module('bridge');c=importlib.import_module('agent_contract');p=importlib.import_module('provider');w=importlib.import_module('tools');r=importlib.import_module('runtime');f=importlib.import_module('fixtures')
finally:sys.path.pop(0)
Agent48=r.Agent;World48=w.World;Fault=b.Fault;digest=b.digest;validate=b.validate;tool_args=c.tool_args;Task=c.Task;Strict=c.Strict;mock=f.mock;task=f.task;TOOLS=c.TOOLS
ReadArgs=c.ReadArgs;SearchArgs=c.SearchArgs;HealthArgs=c.HealthArgs

def source_hash():return digest(dict(base45_bridge46=b.source_hash(),lesson48={n:hashlib.sha256((BASE/(n+'.py')).read_bytes()).hexdigest() for n in NAMES}))
