"""Reuse durable state/journal and planning/MCP gates without editing earlier lessons."""
import importlib,sys,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'51-trang-thai-ben-vung-checkpoint-khoi-phuc'
for n in ['bridge51','contracts51','journal51','runtime51']:
 if not (BASE/(n+'.py')).is_file():raise RuntimeError('keep source45,46,48,49,50,51 beside52')
 if n in sys.modules and Path(sys.modules[n].__file__).resolve()!=BASE/(n+'.py'):raise RuntimeError('module collision; fresh process required')
sys.path.insert(0,str(BASE))
try:
 b=importlib.import_module('bridge51');c=importlib.import_module('contracts51');j=importlib.import_module('journal51');r=importlib.import_module('runtime51')
finally:sys.path.pop(0)
digest=b.digest;loads=b.loads;Strict=b.Strict;Fault=b.Fault;Link=b.Link;MCPWorld=b.MCPWorld;World48=b.World48;task=b.task
Journal=j.Journal;JournalError=j.JournalError;DurableAgent=r.DurableAgent;restore=r.restore;probe_append=r.probe_append
State=c.State;Limits=c.Limits;FINAL=c.FINAL;canonical=c.canonical;operation_id=c.operation_id
import copy
TOOLS=copy.deepcopy(sys.modules['agent_contract'].TOOLS)
next(t for t in TOOLS if t['name']=='search_runbooks')['description']='Search currently allowed runbooks. Empty is not completion; a bounded new query may use actually observed symptoms.'
PlannerScript=b.PlannerScript;PlanError=b.PlanError;verify=b.verify;Terminal=b.Terminal;pc=b.pc

def source_hash():return digest(dict(base50=b.source_hash(),implementation51=r.implementation_hash()))
