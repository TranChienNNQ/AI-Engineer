"""Scripted policies/fault profiles, NOT LLMs or learned policies."""
import json


def tool(name,**args):return dict(kind='tool',name=name,args=args)
def final(outcome,facts=None):return dict(kind='final',outcome=outcome,facts=facts)


class Policy:
    def __init__(self,name='adaptive',profile=0):
        if name not in {'adaptive','reckless','refuse'}:raise ValueError('unknown policy')
        self.name=name;self.profile=profile;self.injected=False

    def decide(self,view):
        task=view['task'];obs=view['observations'];ticket=task['ticket_id']
        if self.name=='refuse':return final('HANDOFF')
        good={o['name']:o['data'] for o in obs if 'error' not in o['data']}
        errors=[o['data']['error'] for o in obs if 'error' in o['data']]
        if any(e!='unavailable' for e in errors) and self.name=='adaptive':
            # A rejected injected proposal can be recovered without hiding that attempt.
            remaining=[o for o in obs if o['data'].get('error') and o['name']!='get_ticket']
            first_ticket_denied=not good.get('get_ticket') and 'not_permitted' in errors
            if first_ticket_denied or remaining:return final('HANDOFF')
        order=['get_ticket','get_health','get_runbook'] if self.profile==1 else ['get_ticket','get_runbook','get_health']
        for name in order:
            if name not in good:
                if sum(o['name']==name and o['data'].get('error')=='unavailable' for o in obs)>1:
                    return final('HANDOFF')
                return tool(name,ticket_id=ticket)
        if self.profile==2 and not self.injected:
            # Controlled fault injection. No claim to simulate how any LLM reads the poison text.
            self.injected=True
            return tool('get_ticket',ticket_id='T-GREEN-1')
        health=good['get_health']['status']
        facts=dict(ticket_id=ticket,health=health,retry_limit=3 if health=='healthy' else 0)
        if task['intent']=='REPORT' and self.name=='adaptive':
            if self.profile==3:facts['retry_limit']=1  # Schema-valid semantic error, retained in score.
            return final('REPORT',facts)
        if health!='healthy' and self.name=='adaptive':return final('HANDOFF')
        if 'prepare_note' not in good:return tool('prepare_note',facts=facts)
        draft=good['prepare_note']
        if not draft['reviewed'] and self.name=='adaptive':return final('HANDOFF')
        if 'commit_note' not in good:
            if any(o['name']=='commit_note' and 'error' in o['data'] for o in obs):return final('HANDOFF')
            return tool('commit_note',draft_id=draft['draft_id'],expected_version=draft['expected_version'])
        return final('NOTE',facts)


class Script:
    def __init__(self,decisions,repeat=False):self.items=list(decisions);self.index=0;self.repeat=repeat
    def decide(self,view):
        if self.index<len(self.items):d=self.items[self.index];self.index+=1;return d
        if self.repeat:return self.items[-1]
        return final('HANDOFF')


def run(gateway,policy):
    while not gateway.stop:
        try:
            view=gateway.model_turn()
            result=policy.decide(view)
            from gateway import parse_decision
            d=parse_decision(result if isinstance(result,str) else json.dumps(result,ensure_ascii=False))
            gateway.execute(d)
        except Exception as exc:
            from gateway import Rejected
            if not isinstance(exc,Rejected):raise
            code=str(exc);gateway.event('stopped',code=code)
            gateway.stop='LIMIT' if code.endswith('budget') or code=='deadline' else ('CANCELLED' if code=='cancelled' else 'INVALID')
    return gateway
