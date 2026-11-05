"""Decoding, context and explicit one-block KV cache. CPU; no downloads or training."""
import argparse
import codecs
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import platform
import torch
from torch.nn import functional as F
from core import ByteBPE, LanguageModel, Config, PAD, BOS, EOS


def distribution(logits, temperature=1., top_k=0, top_p=1.):
    """1-D CPU distribution; temperature -> exact-k -> minimal nucleus -> renormalize."""
    if logits.ndim!=1 or not logits.is_floating_point():raise ValueError('Expected 1-D floating logits')
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Temperature must be positive; greedy is separate')
    if type(top_k) is not int or not 0<=top_k<=logits.numel():raise ValueError('top_k outside vocabulary')
    if not math.isfinite(top_p) or not 0<top_p<=1:raise ValueError('top_p must be in (0,1]')
    if logits.isnan().any() or logits.isposinf().any() or not logits.isfinite().any():raise ValueError('Invalid/all-blocked logits')
    z=logits.clone()/temperature
    if not z.isfinite().any() or z.isposinf().any():raise ValueError('Temperature overflow')
    order=torch.argsort(z,descending=True,stable=True)  # Ties: smaller ID first.
    if top_k:z[order[top_k:]]=-float('inf')
    p=z.softmax(-1)
    if top_p<1:
        order=torch.argsort(p,descending=True,stable=True)
        sorted_p=p[order]
        # Include the token crossing the threshold, plus the first token always.
        keep=sorted_p.cumsum(0)-sorted_p<top_p
        keep[0]=True
        p[order[~keep]]=0
        p=p/p.sum()
    return p


def choose(logits,strategy='greedy',generator=None,**kwargs):
    if strategy=='greedy':
        if logits.isnan().any() or logits.isposinf().any() or not logits.isfinite().any():raise ValueError('Invalid logits')
        return int(logits.argmax())
    if strategy!='sample':raise ValueError('Unknown strategy')
    return int(torch.multinomial(distribution(logits,**kwargs),1,generator=generator))


def allowed_mask(past,new,device=None):
    """True=allowed for SDPA. Query absolute positions past..past+new-1."""
    return torch.arange(past+new,device=device)[None,:] <= (past+torch.arange(new,device=device))[:,None]


@torch.no_grad()
def cached_forward(model,ids,cache=None):
    """One-block pre-LN core only; B=1, unpadded CPU, eval, absolute position offset."""
    if model.training:raise ValueError('Cache requires eval mode')
    if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<1 or ids.eq(PAD).any():raise ValueError('Expected nonempty unpadded [1,L]')
    if cache is not None and cache['owner']!=id(model):raise ValueError('Cache belongs to another model')
    past=0 if cache is None else cache['length']
    if cache is None and int(ids[0,0])!=BOS:raise ValueError('Prefill must start with BOS')
    n=ids.shape[1]
    if past+n>model.cfg.max_len:raise ValueError('Context limit exceeded')
    if cache is not None and cache['k'].shape[-2]!=past:raise ValueError('Invalid cache length')
    d=model.cfg.dim;h=model.cfg.heads;dh=d//h
    x=model.embedding(ids)*math.sqrt(d)+model.position(torch.arange(past,past+n))
    z=model.block.n1(x)
    q,k,v=F.linear(z,model.block.attn.in_proj_weight,model.block.attn.in_proj_bias).chunk(3,-1)
    def split(t):return t.reshape(1,n,h,dh).transpose(1,2)
    q,k,v=map(split,(q,k,v))
    if cache is not None:k=torch.cat([cache['k'],k],-2);v=torch.cat([cache['v'],v],-2)
    # Explicit offset mask works for prefill, one-token decoding AND multi-token chunks.
    a=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed_mask(past,n),dropout_p=0.,is_causal=False)
    a=a.transpose(1,2).reshape(1,n,d)
    x=x+model.block.attn.out_proj(a)
    x=x+model.block.ff(model.block.n2(x))
    logits=model.head(model.norm(x))
    return logits,dict(k=k,v=v,length=past+n,owner=id(model))


def memory_bytes(cache):
    return sum(cache[name].numel()*cache[name].element_size() for name in ['k','v'])


def context_budget(limit,prompt_tokens,max_new_tokens):
    """Conservative serving policy: reserve room for entire prompt + generated output."""
    if any(type(x) is not int for x in [limit,prompt_tokens,max_new_tokens]):raise ValueError('Use integer token counts')
    if limit<1 or prompt_tokens<1 or max_new_tokens<1 or prompt_tokens+max_new_tokens>limit:
        raise ValueError('Prompt plus reserved output exceeds context budget')
    return limit-prompt_tokens-max_new_tokens


@torch.no_grad()
def generate(model,tok,prompt,max_new_tokens=10,strategy='greedy',use_cache=True,seed=21,temperature=1.,top_k=0,top_p=1.):
    model.eval();prefix=[BOS]+tok.encode(prompt)
    context_budget(model.cfg.max_len,len(prefix),max_new_tokens)
    rng=torch.Generator().manual_seed(seed)
    ids=torch.tensor([prefix]);cache=None;output=[];reason='length';processed=0
    for step in range(max_new_tokens):
        if use_cache:
            current=ids if cache is None else ids[:,-1:]
            logits,cache=cached_forward(model,current,cache);processed+=current.shape[1]
        else:
            logits=model(ids);processed+=ids.shape[1]
        scores=logits[0,-1].clone();scores[PAD]=scores[BOS]=-float('inf')
        token=choose(scores,strategy,rng,temperature=temperature,top_k=top_k,top_p=top_p)
        output.append(token)
        if token==EOS:reason='eos';break
        ids=torch.cat([ids,torch.tensor([[token]])],1)
    raw=tok.raw(output)
    try:text=raw.decode('utf-8');valid=True
    except UnicodeDecodeError:text=raw.decode('utf-8',errors='replace');valid=False
    return dict(ids=output,text=text,finish_reason=reason,valid_utf8=valid,prompt_tokens=len(prefix),
                output_tokens=len(output),positions_processed=processed,kv_bytes=memory_bytes(cache) if cache else 0)


def next_toy(prefix):
    """Finite tree: three generated tokens including EOS; probabilities sum to one."""
    if not prefix:return {'A':.6,'B':.4}
    if prefix==('A',):return {f'X{i}':.1 for i in range(10)}
    if prefix==('B',):return {'X0':.9,'X1':.1}
    if len(prefix)==2:return {'EOS':1.}
    raise ValueError('No expansion after completion')


def beam_search(next_probs,width=2,max_new=3):
    if width<1 or max_new<1:raise ValueError('Positive beam width and budget required')
    beams=[((),0.,False)]
    for _ in range(max_new):
        candidates=[]
        for prefix,score,done in beams:
            if done:candidates.append((prefix,score,True));continue
            for token,probability in next_probs(prefix).items():
                if probability>0:candidates.append((prefix+(token,),score+math.log(probability),token=='EOS'))
        beams=sorted(candidates,key=lambda b:(-b[1],b[0]))[:width]
        if all(b[2] for b in beams):break
    return [dict(tokens=list(p),log_probability=s,probability=math.exp(s),ended=e) for p,s,e in beams]


def hand_demo():
    p=torch.tensor([.5,.25,.15,.1],dtype=torch.float64);z=p.log()
    result={name:distribution(z,**kwargs).tolist() for name,kwargs in {
        'tau1':{},'tau_half':{'temperature':.5},'tau2':{'temperature':2.},
        'top_k2':{'top_k':2},'top_p_07':{'top_p':.7},'top_p_08':{'top_p':.8},
        'k2_then_p06':{'top_k':2,'top_p':.6}}.items()}
    torch.testing.assert_close(distribution(z,top_k=2),torch.tensor([2/3,1/3,0,0],dtype=z.dtype))
    torch.testing.assert_close(distribution(z,top_p=.8),torch.tensor([5/9,5/18,1/6,0],dtype=z.dtype))
    g=torch.Generator().manual_seed(21)
    draws=torch.multinomial(distribution(z,top_p=.7),10000,replacement=True,generator=g)
    result['sample_counts']=torch.bincount(draws,minlength=4).tolist()
    print(json.dumps(result,indent=2));return result


def beam_demo():
    greedy=beam_search(next_toy,width=1)[0];beam=beam_search(next_toy,width=2)[0]
    exhaustive=[]
    for first in ['A','B']:
        for second,p in next_toy((first,)).items():exhaustive.append((.6 if first=='A' else .4)*p)
    assert math.isclose(greedy['probability'],.06) and math.isclose(beam['probability'],.36)
    assert math.isclose(beam['probability'],max(exhaustive))
    result=dict(greedy=greedy,beam2=beam,exhaustive_best=max(exhaustive))
    print(json.dumps(result,indent=2));return result


def cache_demo():
    torch.manual_seed(21);model=LanguageModel(259,Config()).double().eval()
    ids=torch.tensor([[BOS,100,101,102,103,104,105,106]])
    with torch.no_grad():
        reference=model(ids)
        prefill,cache=cached_forward(model,ids[:,:4]);chunks=[prefill]
        for i in range(4,8):
            logits,cache=cached_forward(model,ids[:,i:i+1],cache);chunks.append(logits)
        output=torch.cat(chunks,1)
        torch.testing.assert_close(output,reference,rtol=1e-10,atol=1e-10)
        # Multi-token chunk: includes offset and excludes future within the new chunk.
        p,c=cached_forward(model,ids[:,:3]);tail,c=cached_forward(model,ids[:,3:],c)
        torch.testing.assert_close(tail,reference[:,3:],rtol=1e-10,atol=1e-10)
        wrong=F.scaled_dot_product_attention(torch.zeros(1,1,1,1,dtype=torch.float64),torch.zeros(1,1,4,1,dtype=torch.float64),
               torch.tensor([1.,2.,3.,4.],dtype=torch.float64).reshape(1,1,4,1),is_causal=True)
        correct=F.scaled_dot_product_attention(torch.zeros(1,1,1,1,dtype=torch.float64),torch.zeros(1,1,4,1,dtype=torch.float64),
               torch.tensor([1.,2.,3.,4.],dtype=torch.float64).reshape(1,1,4,1),attn_mask=allowed_mask(3,1))
        assert wrong.item()==1 and correct.item()==2.5
    result=dict(max_abs_error=(output-reference).abs().max().item(),chunk_max_abs_error=(tail-reference[:,3:]).abs().max().item(),
                k_shape=list(cache['k'].shape),kv_bytes=memory_bytes(cache),wrong_upper_left=wrong.item(),correct_offset=correct.item())
    print(json.dumps(result,indent=2));return result


def latency_summary(start,arrivals,end):
    if end<start or any(t<start or t>end for t in arrivals) or any(b<a for a,b in zip(arrivals,arrivals[1:])):
        raise ValueError('Invalid event timestamps')
    return dict(ttft_ms=(arrivals[0]-start)*1000 if arrivals else None,
                tpot_ms=(arrivals[-1]-arrivals[0])*1000/(len(arrivals)-1) if len(arrivals)>1 else None,
                end_to_end_ms=(end-start)*1000,output_tokens=len(arrivals))


def context_demo():
    remaining=context_budget(100,10+20+40,30)
    assert remaining==0
    try:context_budget(100,71,30)
    except ValueError:pass
    else:raise AssertionError('Overflow accepted')
    # UTF-8 streaming: one character split across two byte tokens.
    decoder=codecs.getincrementaldecoder('utf-8')('strict')
    pieces=[decoder.decode(bytes([195]),final=False),decoder.decode(bytes([169]),final=False),decoder.decode(b'',final=True)]
    assert pieces==['','é','']
    result=dict(context_remaining=remaining,utf8_stream=pieces,
                positions_uncached=sum(16+i for i in range(8)),positions_cached=16+7,
                cache_mib_32layers_8kvheads_128dim_4096tokens_fp16=2*32*1*8*4096*128*2/2**20,
                hypothetical_latency=latency_summary(0.,[.12,.16,.205,.25],.25))
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


def load_run20(run):
    tok=ByteBPE.load(run/'tokenizer.json')
    cp=torch.load(run/'full-sft.pt',map_location='cpu',weights_only=True)
    if cp['tokenizer_hash']!=tok.fingerprint():raise ValueError('Tokenizer mismatch')
    model=LanguageModel(tok.size,Config(**cp['config']));model.load_state_dict(cp['model']);return model.eval(),tok


def inference_demo(run20=None):
    if run20:model,tok=load_run20(run20);origin='lesson20 full-sft checkpoint supplied by user'
    else:
        torch.manual_seed(21);tok=ByteBPE();model=LanguageModel(tok.size,Config()).eval();origin='random weights: correctness only, no language quality claim'
    prompt='Route: service=api op=read event=timeout.\nTeam: '
    results={}
    for name,opts in [('greedy',{}),('sample_tau07',dict(strategy='sample',temperature=.7)),
                      ('sample_k5',dict(strategy='sample',top_k=5)),('sample_p09',dict(strategy='sample',top_p=.9))]:
        cached=generate(model,tok,prompt,use_cache=True,**opts)
        uncached=generate(model,tok,prompt,use_cache=False,**opts)
        assert cached['ids']==uncached['ids']
        repeated=generate(model,tok,prompt,use_cache=True,**opts)
        assert repeated==cached
        results[name]=dict(cached=cached,uncached=uncached)
    result=dict(origin=origin,vocab=tok.size,prompt=prompt,strategies=results,
                metadata=dict(python=platform.python_version(),torch=torch.__version__,device='CPU',threads=1))
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['hand','beam','cache','context','infer','all'],default='hand')
    p.add_argument('--out',type=Path);p.add_argument('--run20',type=Path);a=p.parse_args()
    if a.out and a.out.exists():raise FileExistsError('Choose a new output directory')
    torch.set_num_threads(1);report={}
    for name,fn in dict(hand=hand_demo,beam=beam_demo,cache=cache_demo,context=context_demo).items():
        if a.mode in (name,'all'):report[name]=fn()
    if a.mode in ('infer','all'):report['inference']=inference_demo(a.run20)
    if a.out:
        a.out.mkdir(parents=True,exist_ok=False);(a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print('Output:',a.out)
if __name__=='__main__':main()
