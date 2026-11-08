"""Reuse lesson45 via its existing lesson46 bridge, in a fresh Python process."""
import importlib,sys,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'46-tich-hop-llm-tro-ly-ho-tro-ky-thuat'
if not (BASE/'bridge45.py').is_file():raise RuntimeError('keep lessons 45 and 46 beside lesson 48')
if 'bridge45' in sys.modules and Path(sys.modules['bridge45'].__file__).resolve()!=BASE/'bridge45.py':raise RuntimeError('module collision; use a fresh process')
sys.path.insert(0,str(BASE))
try:b=importlib.import_module('bridge45')
finally:sys.path.pop(0)
Engine45=b.Engine45;Fault=b.Fault;digest=b.digest;documents=b.documents;TOKENS=b.TOKENS;retrieve=b.retrieve;validate=b.validate

def source_hash():return digest(dict(core45=b.source_hash(),bridge46=hashlib.sha256((BASE/'bridge45.py').read_bytes()).hexdigest()))
