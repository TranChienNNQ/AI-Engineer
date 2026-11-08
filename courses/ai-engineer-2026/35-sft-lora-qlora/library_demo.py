"""Optional real PEFT CPU Linear example; validates API, adapters and config, not LLM/GPU QLoRA."""
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from importlib.metadata import version
import torch
from torch import nn


def run():
    os.environ['HF_HUB_OFFLINE']='1'
    from peft import LoraConfig,get_peft_model,PeftModel
    from transformers import BitsAndBytesConfig,PreTrainedTokenizerFast
    from tokenizers import Tokenizer as Backend,models,pre_tokenizers
    from gpu_recipe import masked_chat,collate
    torch.manual_seed(35);base=nn.Sequential(nn.Linear(3,2,bias=False))
    with torch.no_grad():base[0].weight.copy_(torch.tensor([[1.,0.,-1.],[0.,2.,1.]]))
    model=get_peft_model(deepcopy(base),LoraConfig(r=1,lora_alpha=2,target_modules=['0'],lora_dropout=0.,bias='none'))
    x=torch.tensor([[2.,1.,3.]])
    torch.testing.assert_close(model(x),base(x))
    layer=model.base_model.model[0]
    with torch.no_grad():
        layer.lora_A['default'].weight.copy_(torch.tensor([[1.,0.,1.]]))
        layer.lora_B['default'].weight.copy_(torch.tensor([[.2],[-.1]]))
    torch.testing.assert_close(model(x),torch.tensor([[1.,4.]]))
    loss=.5*(model(x)-torch.tensor([[1.,2.]])).square().sum();loss.backward()
    torch.testing.assert_close(layer.lora_A['default'].weight.grad,torch.tensor([[-.8,-.4,-1.2]]))
    torch.testing.assert_close(layer.lora_B['default'].weight.grad,torch.tensor([[0.],[20.]]))
    opt=torch.optim.SGD((p for p in model.parameters() if p.requires_grad),lr=.001);opt.step();model.eval()
    assert layer.base_layer.weight.grad is None
    with TemporaryDirectory(prefix='lesson35-peft-') as tmp:
        model.save_pretrained(tmp)
        loaded=PeftModel.from_pretrained(deepcopy(base),tmp).eval()
        torch.testing.assert_close(model(x),loaded(x),rtol=0,atol=0)
        dense=loaded.merge_and_unload();torch.testing.assert_close(model(x),dense(x))
        names=sorted(p.name for p in Path(tmp).iterdir())
    config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    backend=Backend(models.WordLevel({'[UNK]':0,'<eos>':2,'system':3,'user':4,'assistant':5,'NET':6},unk_token='[UNK]'))
    backend.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',eos_token='<eos>',pad_token='<eos>')
    tokenizer.chat_template="{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' <eos> ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    ids,labels=masked_chat(tokenizer,'route timeout','NET')
    assert [v for v in labels if v!=-100]==[6,2]
    _,padded_labels=collate([(ids,labels),([4,2],[-100,2])],2)
    assert padded_labels[1,1].item()==2 and padded_labels[1,2:].eq(-100).all()
    return dict(kind='PEFT CPU Linear train/save/load/merge, no pretrained model/download',
                versions={name:version(name) for name in ['torch','peft','transformers','bitsandbytes','accelerate','safetensors']},
                trainable=5,hand_forward=[1.,4.],adapter_roundtrip=True,merge_close=True,saved_files=names,
                config_checked=dict(load_in_4bit=config.load_in_4bit,quant_type=config.bnb_4bit_quant_type,
                                    double_quant=config.bnb_4bit_use_double_quant,compute_dtype=str(config.bnb_4bit_compute_dtype)),
                actual_hf_template_helper_checked=True,gpu_qlora_trained=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
