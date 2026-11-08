"""Reuse the MCP execution/final gates, add planning as a separate bounded layer."""
import hashlib,importlib,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'49-thiet-ke-tool-tich-hop-mcp'
NAMES=['bridge49','contracts49','mcp_server','mcp_link','gateway']
for n in NAMES:
 if not (BASE/(n+'.py')).is_file():raise RuntimeError('keep lessons45,46,48,49 beside lesson50')
 if n in sys.modules and Path(sys.modules[n].__file__).resolve()!=BASE/(n+'.py'):raise RuntimeError('module collision; use fresh Python process')
sys.path.insert(0,str(BASE))
try:
 b=importlib.import_module('bridge49');g=importlib.import_module('gateway');link=importlib.import_module('mcp_link')
finally:sys.path.pop(0)
MCPAgent=g.MCPAgent;MCPWorld=g.MCPWorld;Link=link.Link;Strict=b.Strict;Fault=b.Fault;digest=b.digest;task=b.task;tool_args=b.tool_args
World48=b.World48;ProviderError=sys.modules['provider'].ProviderError;terminal=b.f.terminal;loads=b.c.loads

def source_hash():return digest(dict(base48=b.source_hash(),lesson49={n:hashlib.sha256((BASE/(n+'.py')).read_bytes()).hexdigest() for n in NAMES}))
