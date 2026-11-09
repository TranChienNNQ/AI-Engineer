"""Reuse lesson52 investigation gates; write workflow is a separate guarded subsystem."""
import importlib,sys,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'52-context-engineering-memory-agent'
for n in ['bridge52','contracts52','memory52','context52','runtime52']:
 if not (BASE/(n+'.py')).is_file():raise RuntimeError('keep source45,46,48,49,50,51,52 beside53')
 if n in sys.modules and Path(sys.modules[n].__file__).resolve()!=BASE/(n+'.py'):raise RuntimeError('module collision; fresh process required')
sys.path.insert(0,str(BASE))
try:
 b=importlib.import_module('bridge52');mem=importlib.import_module('memory52');ctx=importlib.import_module('context52');rt=importlib.import_module('runtime52')
finally:sys.path.pop(0)
Strict=b.Strict;Fault=b.Fault;digest=b.digest;canonical=b.canonical;loads=b.loads;World48=b.World48;Link=b.Link;MCPWorld=b.MCPWorld;Journal=b.Journal
MemoryStore=mem.MemoryStore;MemoryError=mem.MemoryError;ContextAgent=rt.ContextAgent

def source_hash():return digest(dict(base51=b.source_hash(),implementation52=rt.implementation_hash()))
