"""A fully offline model-call/tool-result round trip. Not an LLM implementation or provider SDK."""
import copy
import json
from backend import Backend,ToolError
from contracts import call
from workflow import Runtime


class ScriptedModel:
    """Two scripted responses: propose a read, then consume its actual result."""
    def respond(self,messages):
        if messages[-1]['role']=='user':
            c=call('model-1','get_ticket',ticket_id='T-BLUE-1')
            return dict(kind='calls',raw=json.dumps([c]))
        result=messages[-1]['content'][0]
        if result['status']=='error':return dict(kind='final',text='Chưa đọc được ticket: '+result['error'])
        t=result['data']
        return dict(kind='final',text=f"{t['ticket_id']} thuộc service {t['service']}, version {t['version']}. Chưa ghi thay đổi.")


def tool_roundtrip(model=None,backend=None,max_turns=3):
    if type(max_turns) is not int or max_turns<1:raise ValueError('positive_turn_budget_required')
    model=model or ScriptedModel();runtime=Runtime(backend or Backend())
    messages=[dict(role='user',content='Đọc ticket T-BLUE-1 cho tôi.')]
    for turn in range(1,max_turns+1):
        # Copy so the adapter cannot mutate host-owned history through this reference.
        response=model.respond(copy.deepcopy(messages))
        if not isinstance(response,dict):raise ToolError('invalid_model_turn')
        if response.get('kind')=='final' and set(response)=={'kind','text'} and isinstance(response['text'],str):
            return dict(turns=turn,answer=response['text'],messages=messages,calls=runtime.calls,effects=runtime.backend.effects)
        if response.get('kind')!='calls' or set(response)!={'kind','raw'}:raise ToolError('invalid_model_turn')
        results=runtime.read_batch(response['raw'])  # Read-only loop; writes use the reviewed workflow.
        messages.append(dict(role='assistant',content=response))
        messages.append(dict(role='tool',content=results))
    raise ToolError('turn_budget_exhausted')
