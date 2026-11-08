"""LLM adapter seam; keep lesson 45 review/version/receipt gateway unchanged."""
import hashlib
from pathlib import Path
from bridge45 import Engine45,Fault,RunIn,digest,retrieve,validate,source_hash,create_app
from contracts import prompt_input,parse,ground,identity
from llm_adapter import ProviderError


def implementation_hash():
    root=Path(__file__).resolve().parent
    return digest({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['bridge45.py','contracts.py','llm_adapter.py','support.py']})


class SupportEngine(Engine45):
    def __init__(self,path,adapter,policy='exact_quote',**kwargs):
        if policy not in ['exact_quote','citation_only']:raise ValueError('support policy')
        super().__init__(path,**kwargs);self.adapter=adapter;self.policy=policy
        self.manifest.update(answer='llm-claims-draft-v1',model_adapter=adapter.identity(),contracts=identity(),support_policy=policy,base45_source=source_hash(),implementation_hash=implementation_hash(),schema='46.0')
        self.release=digest(self.manifest)[:16]

    def close(self):self.adapter.close();super().close()

    def integrity(self):
        super().integrity()
        if implementation_hash()!=self.manifest['implementation_hash'] or self.policy!=self.manifest['support_policy'] or self.adapter.identity()!=self.manifest['model_adapter'] or identity()!=self.manifest['contracts'] or source_hash()!=self.manifest['base45_source']:raise Fault(503,'llm_release_integrity_failed')

    def ask(self,token,request):
        with self.store.lock:
            req=RunIn.model_validate(request).model_dump();p=self.principal(token);self.integrity();action=req['action']
            if action!='ask' and p['role']!='agent':raise Fault(403,'agent_required')
            id_,old=self.store.begin(p,req['request_key'],digest(dict(request=req,release=self.release)))
            if old:self.revalidate_result(old,p);return old
            trace=[];payload=None;llm=None;stats={}
            def stage(name):
                if len(trace)>=req['max_steps']:raise Fault(429,'step_budget_exhausted')
                trace.append(dict(stage=name,status='ok'))
            def empty(state,code,status):return dict(state=state,code=code,http_status=status,answer='',citations=[])
            try:
                stage('authorize')
                if action in ['note','resolve']:
                    t=self.store.ticket(req['ticket_id'],p)
                    if t['version']!=req['expected_version']:raise Fault(409,'stale_ticket_version')
                    if t['status']!='open':raise Fault(409,'ticket_closed')
                stage('retrieve')
                if self.down:raise Fault(503,'retrieval_unavailable')
                evidence,stats=retrieve(self.docs,p,req['service'],req['message'],self.manifest['min_terms'],self.manifest['context_chars'],self.revoked)
                if not evidence:
                    stage('validate');result=empty('abstained','insufficient_evidence',200)
                else:
                    if req['max_steps']<(4 if action=='ask' else 5):raise Fault(429,'step_budget_exhausted')
                    if not validate(evidence,evidence,self.docs,p,req['service'],self.revoked):raise Fault(409,'evidence_no_longer_allowed')
                    stage('generate');text,llm=self.adapter.generate(prompt_input(req,evidence,self.policy))
                    stage('validate');reply=parse(text);p=self.principal(token)
                    if action!='ask' and p['role']!='agent':raise Fault(403,'agent_required')
                    self.integrity();validated=ground(reply,evidence,self.docs,p,req,self.revoked,self.policy)
                    if reply.decision=='abstain':result=empty('abstained','model_abstained',200)
                    else:
                        result=dict(state='answered',code='ok',http_status=200,answer=validated['answer'],citations=validated['citations'])
                        if action!='ask':
                            stage('propose');draft=validated['draft']
                            payload=dict(operation=action,ticket_id=req['ticket_id'],expected_version=req['expected_version'],title=draft['title'],body=draft['body'],service=req['service'],citations=validated['citations'],release=self.release)
                            result.update(state='proposed',code='review_required')
                    stats['semantic_check']=validated['semantic']
            except ProviderError as error:
                llm=error.meta;result=empty('abstained' if error.code=='model_refusal' else 'failed',error.code,200 if error.code=='model_refusal' else 503)
                if trace:trace[-1]['status']='failed'
            except Fault as error:
                result=empty('failed',error.code,error.status)
                if trace and error.code!='step_budget_exhausted':trace[-1]['status']='failed'
            result.update(release=self.release,service=req['service'],trace=trace,steps=len(trace),proposal_id=None,
                stats=stats|dict(llm=llm,support_policy=self.policy))
            return self.store.finish(id_,result,payload)


def app_for(engine):
    app=create_app(engine);app.title='Lesson 46 LLM support adapter';app.version='46.0';return app
