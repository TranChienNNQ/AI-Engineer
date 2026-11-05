"""Explicit sequential workflow. Bounded retries use modeled time, no sleeping/network."""
import copy
import json
from contracts import call,digest,parse_turn,validate_output,pair_results
from backend import Backend,ToolError


class Runtime:
    def __init__(self,backend,subject='ops_blue',max_calls=10,budget_ms=1000,max_attempts=2):
        if any(type(x) is not int or x<1 for x in [max_calls,budget_ms,max_attempts]):raise ValueError('positive_budgets_required')
        self.backend=backend;self.subject=subject;self.max_calls=max_calls;self.budget_ms=budget_ms
        self.max_attempts=max_attempts;self.calls=0;self.clock=0;self.trace=[];self.seen=set()

    def spend(self,ms):
        if self.clock+ms>self.budget_ms:raise ToolError('budget_exhausted')
        self.clock+=ms

    def invoke(self,c,*,key=None,grant=None):
        # Validate the complete call before dispatch. IDs are protocol identities, not dedup keys.
        try:c=parse_turn(json.dumps([c],ensure_ascii=False))[0]
        except (ValueError,TypeError) as exc:raise ToolError('invalid_proposal') from exc
        if c['id'] in self.seen:raise ToolError('duplicate_call_id')
        self.seen.add(c['id']);attempts=self.max_attempts if c['name']!='append_note' else 1
        for attempt in range(1,attempts+1):
            if self.calls>=self.max_calls:raise ToolError('call_budget_exhausted')
            # Admit only an attempt whose modeled duration fits. Real client needs its own timeout.
            pending=self.backend.faults.get(c['name'],[])
            fault=pending[0] if pending else None
            duration=fault.elapsed_ms if fault and fault.code in {'unavailable','timeout_before'} else self.backend.latency[c['name']]
            if self.clock+duration>self.budget_ms:raise ToolError('budget_exhausted')
            self.calls+=1;start=self.clock
            try:
                data,elapsed=self.backend.execute(self.subject,c,key=key,grant=grant,now=start)
                self.spend(elapsed)
                if not validate_output(c['name'],data):raise ToolError('invalid_output',uncertain=c['name']=='append_note')
                expected=c['arguments'].get('ticket_id')
                if expected and data.get('ticket_id')!=expected:raise ToolError('wrong_result_resource',uncertain=c['name']=='append_note')
                if c['name']=='get_health' and data['service']!=c['arguments']['service']:raise ToolError('wrong_result_resource')
                result=dict(call_id=c['id'],name=c['name'],status='ok',data=data,error=None)
                self.trace.append(dict(call_id=c['id'],tool=c['name'],attempt=attempt,start_ms=start,end_ms=self.clock,status='ok'))
                return result
            except ToolError as exc:
                if self.clock==start:self.spend(exc.elapsed_ms)
                self.trace.append(dict(call_id=c['id'],tool=c['name'],attempt=attempt,start_ms=start,end_ms=self.clock,
                                       status='unknown' if exc.uncertain else 'error',code=exc.code))
                if exc.uncertain or not exc.retryable or attempt==attempts:raise
                self.spend(50*2**(attempt-1))  # Deterministic teaching backoff; production may add jitter.

    def read_batch(self,raw):
        # Parse/validate all first; reject mixed write batches before any dispatch.
        calls=parse_turn(raw)
        if any(c['name']=='append_note' for c in calls):raise ToolError('read_batch_only')
        if any(c['id'] in self.seen for c in calls):raise ToolError('duplicate_call_id')
        results=[]
        for c in calls:
            try:results.append(self.invoke(c))
            except ToolError as e:results.append(dict(call_id=c['id'],name=c['name'],status='error',data=None,error=e.code))
        return pair_results(calls,results)


class Workflow:
    """Created/owned by trusted controller, not constructed from model-generated run state."""
    def __init__(self,backend=None,subject='ops_blue',ticket_id='T-BLUE-1',run_id='run-27',**budgets):
        if not run_id:raise ValueError('run_id_required')
        self.backend=backend or Backend();self.runtime=Runtime(self.backend,subject,**budgets)
        self.subject=subject;self.ticket_id=ticket_id;self.run_id=run_id
        self.state='NEW';self.events=['NEW'];self.context={};self.write_call=None
        self.source_refs=[];self.review=None;self.key=None;self.result=None;self.reason=None

    def transition(self,state):
        self.state=state;self.events.append(state)

    def prepare(self):
        if self.state!='NEW':raise ToolError('invalid_state')
        try:
            self.transition('READING')
            ticket=self.runtime.invoke(call('c-ticket','get_ticket',ticket_id=self.ticket_id))['data']
            self.context['ticket']=ticket
            if ticket['category']!='technical':
                self.reason='route_to_human';self.transition('NEEDS_HUMAN');return self.snapshot()
            kb=self.runtime.invoke(call('c-kb','lookup_runbook',service=ticket['service']))['data']
            if not kb['sources']:
                self.reason='missing_required_evidence';self.transition('NEEDS_HUMAN');return self.snapshot()
            self.context['kb']=kb
            try:health=self.runtime.invoke(call('c-health','get_health',service=ticket['service']))['data']
            except ToolError as exc:
                if exc.code not in {'unavailable','timeout_before'}:raise
                health=dict(service=ticket['service'],state='unknown')
                self.context['health_error']=exc.code
            self.context['health']=health;self.runtime.spend(50)
            refs=[dict(id=s['id'],version=s['version'],body_hash=digest(s['text'])) for s in kb['sources']]
            self.source_refs=refs
            # Scripted template replaces a draft LLM. Data text is never dispatched as another tool.
            note=f"Triage {ticket['service']}; health={health['state']}; nguồn="+','.join(s['id'] for s in kb['sources'])+'. Cần kỹ sư kiểm tra nguyên nhân; chưa kết luận đã khắc phục.'
            self.write_call=call('c-write','append_note',ticket_id=self.ticket_id,expected_version=ticket['version'],note=note)
            self.key=digest([self.run_id,self.subject,'append_note',self.ticket_id])
            self.transition('WAITING_REVIEW')
        except ToolError as exc:
            self.reason=exc.code;self.transition('FAILED')
        return self.snapshot()

    def approve(self,reviewed_call=None,ttl=500):
        """Trusted review event only; the entire note/action is shown to reviewer outside lab."""
        if self.state!='WAITING_REVIEW':raise ToolError('invalid_state')
        reviewed=reviewed_call or self.write_call
        # Correlation id may change without changing business action; full args must match.
        if digest([reviewed['name'],reviewed['arguments']])!=digest([self.write_call['name'],self.write_call['arguments']]):raise ToolError('review_mismatch')
        self.review=self.backend.issue_review(self.subject,reviewed,self.source_refs,now=self.runtime.clock,ttl=ttl)
        self.transition('APPROVED')

    def reject(self):
        if self.state!='WAITING_REVIEW':raise ToolError('invalid_state')
        self.reason='review_rejected';self.transition('CANCELLED')

    def execute(self):
        if self.state!='APPROVED':raise ToolError('invalid_state')
        self.transition('WRITING')
        try:
            self.result=self.runtime.invoke(self.write_call,key=self.key,grant=self.review)['data']
            self.transition('SUCCEEDED')
        except ToolError as exc:
            self.reason=exc.code;self.transition('UNKNOWN' if exc.uncertain else 'FAILED')
        return self.snapshot()

    def reconcile(self):
        if self.state!='UNKNOWN':raise ToolError('invalid_state')
        try:
            result=self.backend.reconcile(self.subject,self.write_call,self.key)
            if result is not None:
                if not validate_output('append_note',result):raise ToolError('invalid_output')
                self.result=result;self.reason=None;self.transition('SUCCEEDED')
        except ToolError as exc:self.reason=exc.code  # Retain UNKNOWN; lack of access is not proof of no effect.
        return self.snapshot()

    def snapshot(self):
        # No review bearer token or credential. Synthetic note/context is research output only.
        return copy.deepcopy(dict(run_id=self.run_id,subject=self.subject,ticket_id=self.ticket_id,state=self.state,
            reason=self.reason,events=self.events,context=self.context,proposal=self.write_call,result=self.result,
            calls=self.runtime.calls,modeled_ms=self.runtime.clock,trace=self.runtime.trace,effects=self.backend.effects))
