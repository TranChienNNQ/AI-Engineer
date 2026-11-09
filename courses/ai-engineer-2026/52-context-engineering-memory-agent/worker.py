import argparse,json
from pathlib import Path
from bridge52 import Link,MCPWorld,Journal,JournalError,Fault
from memory52 import MemoryStore,MemoryError
from context52 import ContextError
from runtime52 import ContextAgent

def work(directory,run_id='run-a',owner='worker1',now=100,pause_after=None,takeover_epoch=None,crash_after_intent=False,create=False,policy_patch=None,before_consume=None):
 d=Path(directory);config=json.loads((d/'config.json').read_text());j=Journal(d/'state.sqlite');m=MemoryStore(d/'memory.sqlite')
 try:
  with Link(d/'world.sqlite') as link:
   w=MCPWorld(d/'world.sqlite',link)
   try:
    a=ContextAgent(j,w,run_id,config['task'],'demo-agent-a',limits=config['limits'],expires_at=160,probe=d/'probe.sqlite',memory=m,policy=config['policy']|(policy_patch or {}),now=now)
    if create:a.create()
    result=a.run(owner,now,takeover_epoch,pause_after,crash_after_intent,before_consume)
   finally:w.close()
  return dict(ok=True,result=result,link_closed=link.closed)
 except (JournalError,MemoryError,ContextError,Fault) as e:return dict(ok=False,error=e.code if isinstance(e,Fault) else str(e))
 finally:j.close();m.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path);p.add_argument('--run-id',default='run-a');p.add_argument('--owner',default='worker1');p.add_argument('--now',type=int,default=100);p.add_argument('--pause-after',type=int);p.add_argument('--takeover-epoch',type=int);p.add_argument('--crash-after-intent',action='store_true');p.add_argument('--create',action='store_true');a=p.parse_args();print(json.dumps(work(**vars(a)),ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
