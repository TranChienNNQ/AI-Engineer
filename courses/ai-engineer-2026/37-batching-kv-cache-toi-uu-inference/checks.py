"""Numerical and lifecycle predicates; explicit loops are independent attention oracles."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from model import Config,make_model,pack,allowed_mask,BOS,PAD,EOS
from generation import generate
from arithmetic import hand,work,kv_bytes,padding
from scheduling import demo,simulate,Job


def run():
    torch.set_num_threads(1);names=[];max_error=0.
    def check(name,condition):
        if not bool(condition):raise AssertionError(name)
        names.append(name)
    def close(name,a,b,tol=2e-6):
        nonlocal max_error
        error=(a-b).abs().max().item();max_error=max(max_error,error);check(name,error<tol)
    def reject(name,fn):
        try:fn()
        except ValueError:check(name,True);return
        raise AssertionError(name+' accepted')
    h=hand();expected=(10*math.e+4)/(2*math.e+1)
    check('hand attention',abs(h['attention']['output']-expected)<1e-12)
    check('work P4 N3',h['work']==dict(uncached_positions=15,cached_positions=6,uncached_causal_pairs=46,cached_causal_pairs=21))
    check('MHA memory 2GiB',h['memory']['mha_4096_bytes']==2*1024**3)
    check('GQA memory .5GiB',h['memory']['gqa_4096_bytes']==512*1024**2)
    check('length bucketing 34 to 24',padding([2,9,3,8])==dict(valid=22,input_order_dense=34,bucketed_dense=24,input_order=[2,9,3,8],bucketed_order=[2,3,8,9]))
    mask=allowed_mask(torch.tensor([3]),torch.tensor([2]),2)[0]
    check('offset causal mask',mask.tolist()==[[True,True,True,True,False],[True,True,True,True,True]])
    # Float64 hand oracle: softmax of [1,0,1] and V=[2,4,8].
    q=torch.tensor([[[[1.]]]],dtype=torch.float64);k=torch.tensor([[[[1.],[0.],[1.]]]],dtype=torch.float64);v=torch.tensor([[[[2.],[4.],[8.]]]],dtype=torch.float64)
    actual=F.scaled_dot_product_attention(q,k,v,attn_mask=torch.ones(1,1,1,3,dtype=torch.bool),dropout_p=0.)
    check('SDPA matches hand',abs(actual.item()-expected)<1e-12)
    wrong=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
    check('non-square causal trap detected',abs(wrong.item()-2)<1e-12 and abs(wrong.item()-expected)>1)
    # Compare all logits for arbitrary fixed tokens; no sampling hides errors.
    for heads in [1,2,4]:
        m=make_model(Config(kv_heads=heads));seq=[[BOS,7,9,6,8,5],[BOS,4,3,11]];ids,lens=pack(seq)
        full=m(ids,lens);sdpa=m(ids,lens,backend='sdpa');close(f'GQA{heads} manual vs sdpa',full,sdpa)
        for row,p in enumerate(seq):
            one,n=pack([p]);close(f'GQA{heads} padded row{row}',full[row,:len(p)],m(one,n)[0])
        c=m.new_cache(['A','B'],16);pref,ll=pack([p[:2] for p in seq]);chunks=[m(pref,ll,c,['A','B'])]
        for j in [2,3]:
            x,n=pack([[p[j]] for p in seq]);chunks.append(m(x,n,c,['A','B']))
        close(f'GQA{heads} token cached logits prefix',torch.cat(chunks,1),full[:,:4])
        check(f'GQA{heads} physical/logical memory',c.reserved_bytes==kv_bytes(2,heads,8,[16,16],4) and c.used_bytes==kv_bytes(2,heads,8,[4,4],4))
        # Ragged chunked prefill then a second chunk; positions differ across rows.
        c=m.new_cache(['A','B'],16);x,n=pack([seq[0][:3],seq[1][:1]]);m(x,n,c,['A','B'])
        x,n=pack([seq[0][3:],seq[1][1:]]);out=m(x,n,c,['A','B'],backend='sdpa')
        close(f'GQA{heads} ragged row A',out[0],full[0,3:])
        close(f'GQA{heads} ragged row B',out[1],full[1,1:len(seq[1])])
        check(f'GQA{heads} ragged lengths',c.lengths.tolist()==[6,4])
        # Row reorder is explicit; compacting preserves each history, with independent storage.
        selected=c.select([1]);before=c.keys[0].clone();x,n=pack([[12]])
        cached=m(x,n,selected,['B']);refids,reflen=pack([seq[1]+[12]])
        close(f'GQA{heads} compacted B history',cached[0,0],m(refids,reflen)[0,-1])
        check(f'GQA{heads} select copies storage',torch.equal(before,c.keys[0]) and c.lengths.tolist()==[6,4])
    m=make_model();c=m.new_cache(['A','B'],16);x,n=pack([[BOS,3],[BOS,4,5]])
    m(x,n,c,['A','B']);before=c.keys[0].clone();old=c.lengths.clone();z,nn=pack([[7],[8]])
    reject('request row mismatch',lambda:m(z,nn,c,['B','A']))
    reject('cache model owner',lambda:make_model()(z,nn,c,['A','B']))
    check('rejected input does not alter cache',torch.equal(before,c.keys[0]) and torch.equal(old,c.lengths))
    cap=m.new_cache(['A'],2);a,b=pack([[BOS,3]]);m(a,b,cap,['A']);reject('overflow',lambda:m(*pack([[4]]),cap,['A']))
    check('overflow lengths unchanged',cap.lengths.tolist()==[2])
    reject('duplicate cache request ids',lambda:m.new_cache(['A','A'],8))
    reject('empty cache selection',lambda:c.select([]))
    reject('duplicate cache selection',lambda:c.select([0,0]))
    reject('cache invalid capacity',lambda:m.new_cache(['A'],129))
    reject('invalid head divisibility',lambda:Config(kv_heads=3))
    reject('empty input',lambda:pack([[]]))
    reject('prefill without BOS',lambda:m(*pack([[7,8]])))
    reject('PAD inside valid input',lambda:m(torch.tensor([[BOS,PAD]]),torch.tensor([2])))
    reject('non-PAD outside lengths',lambda:m(torch.tensor([[BOS,7]]),torch.tensor([1])))
    extra=torch.tensor([[BOS,7,0,0]]);close('extra right padding works',m(extra,torch.tensor([2]))[0,:2],m(*pack([[BOS,7]]))[0])
    reject('budget too long',lambda:generate(m,[[BOS,3]],max_new=127))
    reject('unknown backend',lambda:m(*pack([[BOS]]),backend='fake'))
    reject('invalid vocab id',lambda:m(*pack([[BOS,99]])))
    m.train();reject('train mode forbidden',lambda:m(*pack([[BOS]])));m.eval()
    with torch.no_grad():m.head.weight[0,0]+=1
    reject('weight update invalidates cache',lambda:m(z,nn,c,['A','B']))
    m=make_model();prompts=[[BOS,4,6],[BOS,7,8,9,10],[BOS,3]]
    a=generate(m,prompts,8,True);b=generate(m,prompts,8,False);d=generate(m,prompts,8,True,'sdpa')
    check('greedy cached/full outputs',a['outputs']==b['outputs']==d['outputs'])
    for i,p in enumerate(prompts):check(f'batch vs single greedy row{i}',a['outputs'][i]==generate(m,[p],8)['outputs'][0])
    # Force EOS/budget branches to test lifecycle, not to claim model learned to stop.
    def fixture(logits,step,active):return [EOS if (i==0 or (i==1 and step==2)) else 7 for i in active]
    a=generate(m,prompts,4,True,selector=fixture);b=generate(m,prompts,4,False,selector=fixture)
    check('forced EOS parity',a['outputs']==b['outputs']==[[EOS],[7,7,EOS],[7,7,7,7]])
    check('forced stop reasons',a['stop_reasons']==['eos','eos','length'])
    check('compaction maps original ids',[r['active_ids'] for r in a['steps']]==[['r0','r1','r2'],['r1','r2'],['r1','r2'],['r2']])
    check('cache work equals P+N-1',a['valid_positions_processed']==sum(map(len,prompts))+sum(len(v)-1 for v in a['outputs']))
    check('full work independent sum',b['valid_positions_processed']==sum(sum(len(p)+j for j in range(len(out))) for p,out in zip(prompts,b['outputs'])))
    s=demo();check('scheduling makespan',s['static']['makespan']==6 and s['continuous']['makespan']==4)
    check('scheduling first/finish',[(r['first_token'],r['finish']) for r in s['continuous']['records']]==[(1,4),(1,1),(2,3),(4,4)])
    reject('invalid scheduling job',lambda:simulate([Job('A',0,0)]))
    return dict(passed=len(names),checks=len(names),max_logit_abs_error=max_error,names=names)
