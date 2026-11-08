"""A release pins a complete synthetic model fixture and its actual local source."""
import platform
import sqlite3
from pathlib import Path
from data import datasets,FEATURES
from model import evaluate

ROOT=Path(__file__).resolve().parent
COMPONENTS={'model','preprocessing','contract','train','validation','metrics','source','environment'}


def source_snapshot():
    return {p.name:p.read_text() for p in sorted(ROOT.glob('*.py'))} | {'requirements.txt':(ROOT/'requirements.txt').read_text()}


def environment():
    return dict(python=platform.python_version(),implementation=platform.python_implementation(),
                system=platform.system(),machine=platform.machine(),sqlite=sqlite3.sqlite_version,dependencies='stdlib-only')


def build(registry,release,threshold=.5,mutate=None):
    train,validation=datasets()
    model=dict(format='logistic-v1',weights=[4.0,1.0],bias=-2.0,threshold=threshold)
    preprocessing=dict(feature_order=FEATURES,scale=[1.0,1.0])
    contract=dict(input_schema='ticket-v1',features=FEATURES,output='priority-0-or-1',label_mapping={'0':'normal','1':'urgent'})
    values=dict(model=model,preprocessing=preprocessing,contract=contract,train=train,validation=validation,
                metrics=evaluate(model,preprocessing,validation['rows']),source=source_snapshot(),environment=environment())
    if mutate:mutate(values)
    manifest=dict(format='ticket-release-v1',origin='fixed-weights-synthetic-not-trained',
                  objects={key:registry.put(value) for key,value in values.items()})
    registry.register(release,manifest)
    return manifest


def load(registry,release):
    manifest=registry.manifest(release)
    if set(manifest)!={'format','origin','objects'} or manifest['format']!='ticket-release-v1' or manifest['origin']!='fixed-weights-synthetic-not-trained':raise ValueError('manifest_format')
    if not isinstance(manifest['objects'],dict) or set(manifest['objects'])!=COMPONENTS:raise ValueError('manifest_components')
    return {key:registry.get(value) for key,value in manifest['objects'].items()}


def seed(registry):
    for name,threshold in [('baseline',.6),('candidate',.5),('slice-regression',.3)]:build(registry,name,threshold)
    return ['baseline','candidate','slice-regression']
