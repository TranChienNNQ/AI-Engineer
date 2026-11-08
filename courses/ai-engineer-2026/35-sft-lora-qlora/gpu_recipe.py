"""Optional CUDA QLoRA-style smoke recipe. GPU path NOT executed in the authored CPU environment."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version,PackageNotFoundError
import torch


def preflight():
    packages={}
    for name in ['torch','transformers','peft','bitsandbytes','accelerate']:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    cuda=torch.cuda.is_available()
    return dict(python=platform.python_version(),platform=platform.system(),packages=packages,cuda=cuda,
                bf16_supported=bool(cuda and torch.cuda.is_bf16_supported()),
                scope='Recipe targets a single CUDA device; preflight is not evidence of NF4 training or memory fit')


def masked_chat(tokenizer,prompt,answer,max_len=128):
    messages=[dict(role='system',content='Route the ticket to one team: NET, AUTH, DATA. Return only the team.'),dict(role='user',content=prompt)]
    prefix=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
    ids=tokenizer.apply_chat_template(messages+[dict(role='assistant',content=answer)],tokenize=True,add_generation_prompt=False,return_dict=False)
    if not isinstance(ids,list) or ids[:len(prefix)]!=prefix:raise ValueError('template prefix mismatch; inspect actual assistant boundary')
    if len(ids)>max_len or len(ids)<=len(prefix):raise ValueError('oversize or no supervised completion')
    labels=[-100]*len(prefix)+ids[len(prefix):]
    return ids,labels


def collate(examples,pad_id):
    length=max(len(x) for x,y in examples)
    ids=torch.full((len(examples),length),pad_id,dtype=torch.long);labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
    for i,(x,y) in enumerate(examples):ids[i,:len(x)]=torch.tensor(x);labels[i,:len(y)]=torch.tensor(y);mask[i,:len(x)]=1
    return dict(input_ids=ids,attention_mask=mask),labels


def local_manifest(path):
    files=sorted(p for p in path.rglob('*') if p.is_file())
    if not files:raise ValueError('empty base directory')
    result={}
    for file in files:
        h=hashlib.sha256()
        with file.open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        result[str(file.relative_to(path))]=h.hexdigest()
    return result


def train(base_dir,out,updates=2):
    info=preflight()
    if not info['cuda'] or not info['bf16_supported']:raise RuntimeError('This recipe needs CUDA with BF16; GPU training has not been verified on this host')
    if any(v is None for v in info['packages'].values()):raise RuntimeError('Install optional requirements in a compatible GPU environment')
    if type(updates) is not int or not 1<=updates<=8:raise ValueError('updates1..8')
    base_dir=Path(base_dir).resolve();out=Path(out).resolve()
    if not base_dir.is_dir() or out.exists() or out.is_relative_to(base_dir):raise ValueError('local base and fresh separate output required')
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from data import records
    manifest=local_manifest(base_dir)
    tok=AutoTokenizer.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False)
    if not tok.chat_template or tok.eos_token_id is None:raise ValueError('chat template and EOS required')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False,
                                              quantization_config=quant,device_map={'':0},dtype=torch.bfloat16)
    if not getattr(model,'is_loaded_in_4bit',False):raise RuntimeError('4-bit load not confirmed')
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',target_modules='all-linear',task_type='CAUSAL_LM'))
    params=[p for n,p in model.named_parameters() if p.requires_grad]
    if not params or any(p.requires_grad and 'lora_' not in n for n,p in model.named_parameters()):raise RuntimeError('unexpected trainables')
    # Standard AdamW is deliberate: this smoke recipe does NOT implement paged optimizer behavior.
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.0);rows=records()['train'][:8]
    examples=[masked_chat(tok,r['prompt'],r['answer']) for r in rows];losses=[];model.train()
    for _ in range(updates):
        opt.zero_grad(set_to_none=True);total_tokens=sum(sum(v!=-100 for v in y[1:]) for x,y in examples)
        if not total_tokens:raise ValueError('no shifted target')
        total=0.
        for start in range(0,len(examples),2):
            inputs,labels=collate(examples[start:start+2],tok.pad_token_id)
            inputs={k:v.to('cuda:0') for k,v in inputs.items()};labels=labels.to('cuda:0')
            logits=model(**inputs).logits
            raw=torch.nn.functional.cross_entropy(logits[:,:-1].float().flatten(0,1),labels[:,1:].flatten(),ignore_index=-100,reduction='sum')
            (raw/total_tokens).backward();total+=float(raw.detach())
        torch.nn.utils.clip_grad_norm_(params,1.0);opt.step();losses.append(total/total_tokens)
    out.mkdir(parents=True);model.save_pretrained(out);tok.save_pretrained(out)
    (out/'run-metadata.json').write_text(json.dumps(dict(preflight=info,base_files=manifest,losses=losses,
        updates=updates,rows=len(rows),scope='GPU smoke only, no validation/test or quality claim; not exact resume checkpoint',
        optimizer='standard AdamW, not paged',target_modules='all-linear'),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['preflight','train'],default='preflight')
    p.add_argument('--base-dir',type=Path);p.add_argument('--out',type=Path);p.add_argument('--updates',type=int,default=2);a=p.parse_args()
    if a.mode=='preflight':print(json.dumps(preflight(),indent=2));return
    if a.base_dir is None or a.out is None:p.error('--base-dir and --out required for train')
    train(a.base_dir,a.out,a.updates)


if __name__=='__main__':main()
