"""Mechanism regressions, not large-model quality or GPU QLoRA certification."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import torch
from torch import nn
from data import records,Tokenizer,batch,example,IGNORE,PAD,EOS,BOS,data_hash
from model import TinyLM,LoRALinear,adapt,frozen_state,state_hash,merged
from quantization import pack,unpack,quantize,dequantize,NF4,NF4Linear
from training import loss_sum,save_adapter,load_adapter,adapter_metadata
from demos import hand,memory
from gpu_recipe import collate,masked_chat,preflight


def evaluate():
    torch.set_num_threads(1);torch.manual_seed(3510);checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(dict(name=name,passed=True))
    def rejects(fn):
        try:fn()
        except (ValueError,RuntimeError):return True
        return False
    splits=records();tok=Tokenizer.fit(splits['train']);h=hand()
    check('hand_forward',h['output']==[1.,4.] and h['loss']==2)
    check('hand_gradients',torch.allclose(torch.tensor(h['grad_A']),torch.tensor([[-.8,-.4,-1.2]])) and h['grad_B']==[[0.],[20.]] )
    check('parameter_formula',h['large_matrix']['lora_params']==65536 and h['large_matrix']['full_params']==16777216)
    group_sets=[{tuple(r['group']) for r in splits[k]} for k in ['train','validation','test']]
    check('group_split_disjoint',not any(group_sets[i]&group_sets[j] for i in range(3) for j in range(i)))
    check('split_sizes',[len(splits[k]) for k in ['train','validation','test']]==[96,24,24])
    check('tokenizer_no_unknown_eval',all(5 not in tok.encode(r['prompt'])+tok.encode(r['answer']) for k in ['validation','test'] for r in splits[k]))
    x,y=batch(splits['train'][:2],tok);valid=y.ne(IGNORE)
    check('answer_plus_eos_only',valid.sum(1).tolist()==[2,2] and y[:,-1].eq(EOS).all())
    check('last_prompt_predicts_first_answer',y[0,-2].item()==tok.encode(splits['train'][0]['answer'])[0])
    logits=torch.zeros(*y.shape,tok.size,requires_grad=True)
    raw,n=loss_sum(logits,y);(raw/n).backward()
    check('uniform_sft_loss',abs(float(raw.detach()/n)-math.log(tok.size))<1e-6)
    check('ignored_logit_gradient_zero',logits.grad[~valid].eq(0).all().item())
    check('supervised_logit_gradient_nonzero',logits.grad[valid].abs().sum().item()>0)
    check('all_masked_rejected',rejects(lambda:loss_sum(logits,torch.full_like(y,IGNORE))))
    check('oversize_not_truncated',rejects(lambda:batch(splits['train'][:1],tok,max_len=2)))
    ragged=[splits['train'][0],dict(splits['train'][1],prompt='route')];px,py=batch(ragged,tok)
    check('padding_labels_ignored',py[px.eq(PAD)].eq(IGNORE).all().item())
    model=TinyLM(tok.size);z=model(x);base_prefix=model(x[:,:5])
    check('causal_prefix_invariant',torch.allclose(z[:,:5],base_prefix,atol=1e-6))
    with torch.no_grad():
        single=model(px[1:2,:4]);padded=model(px)[1:2,:4]
    check('right_padding_invariant',torch.allclose(single,padded,atol=1e-6))
    model.zero_grad();loss_sum(model(x),y)[0].backward()
    prompt_token=int(x[0,2]);check('prompt_can_receive_indirect_gradient',model.embedding.weight.grad[prompt_token].abs().sum().item()>0)
    layer=LoRALinear(nn.Linear(3,2,bias=False),rank=1,alpha=2);inp=torch.tensor([[2.,1.,3.]])
    check('zero_B_preserves_base',torch.equal(layer(inp),layer.base(inp)))
    loss=layer(inp).square().sum();loss.backward()
    check('first_A_grad_zero',layer.A.grad.eq(0).all().item())
    check('first_B_grad_nonzero',layer.B.grad.abs().sum().item()>0)
    check('base_gradient_none',layer.base.weight.grad is None)
    both=LoRALinear(nn.Linear(3,2,bias=False),1,2)
    with torch.no_grad():both.A.zero_()
    both(inp).square().sum().backward()
    check('both_zero_initialization_stalls',both.A.grad.eq(0).all().item() and both.B.grad.eq(0).all().item())
    check('invalid_target_rejected',rejects(lambda:adapt(model,targets=['missing'])))
    codes=torch.arange(16,dtype=torch.uint8)
    check('nibble_roundtrip',torch.equal(unpack(pack(codes)),codes))
    check('nibble_size',pack(codes).numel()==8)
    check('invalid_nibbles',rejects(lambda:pack(torch.tensor([16,0],dtype=torch.uint8))))
    zero=torch.zeros(3,5);p,s=quantize(zero,8);d=dequantize(p,s,zero.shape,8)
    check('zero_block_exact',torch.equal(d,zero) and torch.isfinite(d).all().item())
    exact=NF4.reshape(2,8);p,s=quantize(exact,16)
    check('codebook_roundtrip',torch.equal(dequantize(p,s,exact.shape,16),exact))
    check('quantize_invalid_block',rejects(lambda:quantize(zero,3)))
    check('quantize_nan_rejected',rejects(lambda:quantize(torch.tensor([[float('nan')]]))))
    quant=NF4Linear(nn.Linear(3,2,bias=False),8);g=torch.tensor([[1.,2.,3.]],requires_grad=True)
    quant(g).sum().backward()
    check('frozen_quantized_path_keeps_input_grad',g.grad is not None and g.grad.abs().sum().item()>0)
    check('quantized_base_has_no_trainable_weights',not list(quant.parameters()))
    for kind in ['lora','packed-nf4-lora']:
        torch.manual_seed(3511);m=adapt(model,kind=='packed-nf4-lora');before=state_hash(frozen_state(m))
        trainables=[p for p in m.parameters() if p.requires_grad]
        check(kind+'_trainable_count',sum(p.numel() for p in trainables)==1176)
        optimizer=torch.optim.AdamW(trainables,lr=.01,weight_decay=0)
        optimizer.zero_grad();r,n=loss_sum(m(x),y);(r/n).backward();optimizer.step()
        check(kind+'_frozen_unchanged',state_hash(frozen_state(m))==before)
        check(kind+'_merge_forward',torch.allclose(m(x),merged(m)(x),atol=2e-5,rtol=2e-5))
        with TemporaryDirectory(prefix='lesson35-check-') as tmp:
            path=Path(tmp)/'adapter.pt';meta=adapter_metadata(model,tok,kind,data_hash(splits));save_adapter(path,m,meta)
            loaded=load_adapter(path,model,tok,kind,data_hash(splits))
            check(kind+'_roundtrip',torch.equal(m(x),loaded(x)))
            wrong=deepcopy(model)
            with torch.no_grad():wrong.embedding.weight[1,0]+=.01
            check(kind+'_base_mismatch_rejected',rejects(lambda:load_adapter(path,wrong,tok,kind,data_hash(splits))))
            check(kind+'_dataset_mismatch_rejected',rejects(lambda:load_adapter(path,model,tok,kind,'other')))
    # Unequal supervised-token counts: weighted accumulation equals a single full-batch CE.
    a=TinyLM(tok.size);b=deepcopy(a);rows=[splits['train'][0],dict(splits['train'][1],answer='NET NET')]
    xx,yy=batch(rows,tok);raw,n=loss_sum(a(xx),yy);(raw/n).backward()
    for row in rows:
        xx1,yy1=batch([row],tok);raw1,n1=loss_sum(b(xx1),yy1);(raw1/n).backward()
    check('token_weighted_accumulation',all(torch.allclose(p.grad,q.grad,atol=3e-6,rtol=3e-6) for p,q in zip(a.parameters(),b.parameters())))
    class FakeTokenizer:
        def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict):
            assert return_dict is False
            return [1,9,4] if add_generation_prompt else [1,9,4,8,2]
    ids,labels=masked_chat(FakeTokenizer(),'x','NET')
    check('hf_recipe_unshifted_labels',ids==[1,9,4,8,2] and labels==[-100,-100,-100,8,2])
    inputs,labels=collate([(ids,labels),([1,2],[-100,2])],2)
    check('pad_alias_eos_preserves_real_eos',labels[1,1].item()==2 and labels[1,2:].eq(-100).all().item() and inputs['attention_mask'][1].tolist()==[1,1,0,0,0])
    check('memory_not_4bit_everything',memory()['ideal_nf4_with_scales_bits']==4.5 and memory()['ideal_nested_bits']>4)
    check('preflight_reports_scope',preflight()['scope'].startswith('Recipe targets'))
    return dict(kind='SFT/LoRA/packed NF4 mechanism regression; no GPU QLoRA quality claim',checks=len(checks),passed=len(checks),rows=checks)
