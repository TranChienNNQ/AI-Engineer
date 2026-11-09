import argparse,json
from pathlib import Path
from bridge53 import Fault,MemoryError
from workflow53 import Workflow,ReviewError,render_review

def work(directory,action,op_id='note-a',token='demo-agent-a',now=100,expected_revision=None,display_hash=None,comment=None,retry_same_key=False,crash_at=None,fault=None,revoke=False):
 w=Workflow(directory,token,now,fault,revoke)
 try:
  if action=='prepare':result=w.create(op_id)
  elif action=='view':
   result=w.view(op_id);render_review(result,Path(directory)/'review.html')
  elif action in ['approve','reject']:result=w.review(op_id,expected_revision,display_hash,action)
  elif action=='edit':result=w.edit(op_id,expected_revision,comment)
  elif action=='cancel':result=w.cancel(op_id)
  elif action=='execute':result=w.execute(op_id,retry_same_key,crash_at)
  elif action=='reconcile':result=w.reconcile(op_id)
  else:raise ValueError('unknown action')
  # Metadata response for mutations; the review action is the explicit content view.
  if action!='view':result={k:result[k] for k in ['revision','hash']}|{k:result['state'][k] for k in ['phase','attempts','status_checks','receipt','last_code']}
  return dict(ok=True,result=result)
 except (ReviewError,MemoryError,Fault) as e:return dict(ok=False,error=e.code if isinstance(e,Fault) else str(e))
 finally:w.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--action',required=True,choices=['prepare','view','approve','reject','edit','cancel','execute','reconcile']);p.add_argument('--op-id',default='note-a');p.add_argument('--token',default='demo-agent-a');p.add_argument('--now',type=int,default=100);p.add_argument('--expected-revision',type=int);p.add_argument('--display-hash');p.add_argument('--comment');p.add_argument('--retry-same-key',action='store_true');p.add_argument('--crash-at',choices=['before_send','after_response']);p.add_argument('--fault',choices=['before_send','after_commit','before_commit','status_unavailable','status_missing','bad_receipt']);p.add_argument('--revoke',action='store_true');print(json.dumps(work(**vars(p.parse_args())),ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
