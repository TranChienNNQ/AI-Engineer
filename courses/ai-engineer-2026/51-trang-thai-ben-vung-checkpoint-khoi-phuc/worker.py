"""Fresh-process worker. Crash flags are trusted test hooks, never model inputs."""
import argparse,json
from pathlib import Path
from bridge51 import Link,MCPWorld,Fault
from journal51 import Journal,JournalError
from runtime51 import DurableAgent

def work(directory,owner='worker1',takeover_epoch=None,crash_at=None,crash_step='S2',pause_after=None,now=100,deny=False,revoke=False,health='degraded',config_patch=None):
 d=Path(directory);config=json.loads((d/'config.json').read_text());config.update(config_patch or {})
 journal=Journal(d/'state.sqlite');agent=None
 try:
  with Link(d/'world.sqlite',health_status=health) as link:
   world=MCPWorld(d/'world.sqlite',link,health_status=health)
   try:
    if deny:world.core.identities['demo-agent-a']['active']=False
    if revoke:world.core.revoked.add('gateway')
    agent=DurableAgent(journal,world,config['run_id'],config['task'],'demo-agent-a',limits=config['limits'],expires_at=config['expires_at'],probe=d/'probe.sqlite',crash_at=crash_at,crash_step=crash_step,pause_after=pause_after)
    result=agent.run(owner,now,takeover_epoch)
   finally:world.close()
  return dict(ok=True,result=result,link_closed=link.closed)
 except (JournalError,Fault) as e:return dict(ok=False,error=e.code if isinstance(e,Fault) else str(e))
 finally:journal.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path);p.add_argument('--owner',default='worker1');p.add_argument('--takeover-epoch',type=int);p.add_argument('--crash-at',choices=['planning_reserved','intent_committed','attempt_reserved','after_response','observation_committed','terminal_transaction','terminal_committed']);p.add_argument('--crash-step',default='S2');p.add_argument('--pause-after',type=int);p.add_argument('--now',type=int,default=100)
 a=p.parse_args();print(json.dumps(work(**vars(a)),ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
