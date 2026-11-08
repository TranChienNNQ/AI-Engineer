"""Right-padded prefill, per-row last logits, compact completed rows during decode."""
import torch
from model import pack,BOS,EOS


@torch.inference_mode()
def generate(model,prompts,max_new=8,use_cache=True,backend='manual',selector=None):
    if type(max_new) is not int or max_new<1 or not prompts:raise ValueError('positive output budget and nonempty prompts')
    if any(not p or p[0]!=BOS or len(p)+max_new>model.c.max_len for p in prompts):raise ValueError('BOS prompt/context budget')
    ids,lengths=pack(prompts);names=tuple(f'r{i}' for i in range(len(prompts)))
    cache=model.new_cache(names,max(len(p) for p in prompts)+max_new) if use_cache else None
    active=list(range(len(prompts)));seq=[list(p) for p in prompts];outputs=[[] for _ in prompts];reasons=['length']*len(prompts)
    processed_valid=processed_dense=0;steps=[]
    for step in range(max_new):
        if use_cache and step:
            ids,lengths=pack([[outputs[i][-1]] for i in active])
        elif not use_cache:ids,lengths=pack([seq[i] for i in active])
        logits=model(ids,lengths,cache,tuple(names[i] for i in active) if cache else None,backend)
        last=logits[torch.arange(len(active)),lengths-1]
        last=last.clone();last[:,0:2]=-torch.inf  # forbid PAD/BOS in output, consistently across paths
        next_ids=last.argmax(-1).tolist() if selector is None else list(selector(last,step,active))
        if len(next_ids)!=len(active) or any(type(v) is not int or not 0<=v<model.c.vocab or v==0 for v in next_ids):raise ValueError('selector output')
        processed_valid+=int(lengths.sum());processed_dense+=ids.numel();keep=[]
        for row,(i,token) in enumerate(zip(active,next_ids)):
            outputs[i].append(token);seq[i].append(token)
            if token==EOS:reasons[i]='eos'
            elif step+1<max_new:keep.append(row)
        steps.append(dict(step=step+1,active_ids=[names[i] for i in active],cache_lengths=cache.lengths.tolist() if cache else None,
                          reserved_bytes=cache.reserved_bytes if cache else 0,used_bytes=cache.used_bytes if cache else 0))
        if not keep:break
        active=[active[row] for row in keep]
        if cache is not None and len(keep)<len(cache.request_ids):cache=cache.select(keep)
    return dict(outputs=outputs,stop_reasons=reasons,valid_positions_processed=processed_valid,dense_positions_processed=processed_dense,
                output_tokens=sum(map(len,outputs)),steps=steps)
