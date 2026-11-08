"""Reuse lesson50 policy, plan guards and final/MCP gates; introduce a durable loop."""
import importlib,sys,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'50-lap-ke-hoach-kiem-chung-doi-ke-hoach'
NAMES=['bridge50','plan_contract','verification','plan_guard','policy50','planning']
for n in NAMES:
 if not (BASE/(n+'.py')).is_file():raise RuntimeError('keep source45,46,48,49,50 beside51')
 if n in sys.modules and Path(sys.modules[n].__file__).resolve()!=BASE/(n+'.py'):raise RuntimeError('module collision; use fresh process')
sys.path.insert(0,str(BASE))
try:
 b=importlib.import_module('bridge50');pc=importlib.import_module('plan_contract');v=importlib.import_module('verification');pg=importlib.import_module('plan_guard');policy=importlib.import_module('policy50');planning=importlib.import_module('planning')
finally:sys.path.pop(0)
Strict=b.Strict;Fault=b.Fault;digest=b.digest;loads=b.loads;World48=b.World48;Link=b.Link;MCPWorld=b.MCPWorld;task=b.task
PlanGuard=pg.PlanGuard;PlanError=pg.PlanError;PlannerScript=policy.PlannerScript;verify=v.verify;PlannedAgent=planning.PlannedAgent;PlanningPort=planning.PlanningPort
Terminal=b.b.c.Terminal

def source_hash():return digest(dict(base49=b.source_hash(),lesson50=planning.implementation_hash()))
