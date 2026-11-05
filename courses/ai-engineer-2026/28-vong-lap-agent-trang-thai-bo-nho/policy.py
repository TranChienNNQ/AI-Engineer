"""Offline policies, not LLMs. Adapts to observations while host keeps loop/rights/budgets."""
import json
from contracts import tool


class AdaptivePolicy:
    model_id='adaptive-v1'
    def choose(self,view):
        obs=view['observations'];task=view['task'];last=view['recent'][-1] if view['recent'] else {}
        if last.get('type')=='tool_error':return json.dumps(dict(kind='handoff',reason=last['code'] if last['code'] in {'unavailable','not_permitted','outside_task','ticket_required'} else 'need_human'))
        if 'get_ticket' not in obs:return json.dumps(tool('get_ticket',ticket_id=task['ticket_id']))
        ticket=obs['get_ticket']['data']
        if ticket['category']!='technical':return json.dumps(dict(kind='handoff',reason='outside_domain'))
        if 'get_runbook' not in obs:return json.dumps(tool('get_runbook',service=ticket['service']))
        if task['intent']=='triage' and 'get_health' not in obs:return json.dumps(tool('get_health',service=ticket['service']))
        ids=sorted({ref['id'] for o in obs.values() for ref in o['refs']})
        kb=obs['get_runbook']['data'];health=obs.get('get_health',{}).get('data',{}).get('state','not_requested')
        text=f"{ticket['service']}: retry {','.join(kb['retryable'])}; không retry {','.join(kb['not_retry'])}; tối đa {kb['max_attempts']} attempt. Health={health}. Chưa xác nhận root cause."
        return json.dumps(dict(kind='final',answer=text,source_ids=ids),ensure_ascii=False)


class RepeatPolicy:
    model_id='repeat-v1'
    def choose(self,view):return json.dumps(tool('get_ticket',ticket_id=view['task']['ticket_id']))


class ScriptPolicy:
    model_id='script-v1'
    def __init__(self,items):self.items=items
    def choose(self,view):
        # Context-derived turn index survives rebuilding this object after resume.
        index=view['server']['turn_index']
        item=self.items[min(index,len(self.items)-1)]
        return item if isinstance(item,str) else json.dumps(item,ensure_ascii=False)

POLICIES={'adaptive-v1':AdaptivePolicy,'repeat-v1':RepeatPolicy}
