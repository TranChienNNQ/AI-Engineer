import copy,re,time
from corpus import documents,TOKENS,digest
from retrieval import retrieve,validate
from store import Store,Fault


class Engine:
    def __init__(self,path,docs=None,min_terms=2,clock=time.time):
        if min_terms not in [1,2]:raise ValueError('supported overlap configurations: 1/2')
        self.store=Store(path);self.docs=copy.deepcopy(documents() if docs is None else docs);self.identities=copy.deepcopy(TOKENS);self.revoked=set();self.clock=clock
        self.manifest=dict(corpus_hash=digest(self.docs),retriever='bm25-eligible-v1',min_terms=min_terms,top_k=2,context_chars=600,answer='extract-first-paragraph-v1',policy='review-before-local-write-v1',schema='45.0')
        self.release=digest(self.manifest)[:16];self.down=False;self.corrupt=False

    def close(self):self.store.close()

    def principal(self,token):
        p=self.identities.get(token)
        if not p or not p['active']:raise Fault(401,'unauthorized')
        return dict(p)

    def authenticate(self,p):
        if p not in self.identities.values() or not p.get('active'):raise Fault(401,'unauthorized')

    def integrity(self):
        if digest(self.docs)!=self.manifest['corpus_hash']:raise Fault(503,'release_integrity_failed')

    def ask(self,token,request):
        with self.store.lock:
            p=self.principal(token);self.integrity()
            required={'request_key','action','service','message','ticket_id','expected_version','max_steps'}
            if set(request)!=required or request['action'] not in ['ask','create','note','resolve'] or request['service'] not in ['gateway','upload','identity','database']:raise Fault(422,'invalid_request')
            if not isinstance(request['message'],str) or not 1<=len(request['message'].strip())<=1000 or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',request['request_key']) or type(request['max_steps']) is not int or not 1<=request['max_steps']<=10:raise Fault(422,'invalid_request')
            action=request['action']
            if action in ['note','resolve']:
                if type(request['ticket_id']) is not int or type(request['expected_version']) is not int or request['expected_version']<1:raise Fault(422,'ticket_version_required')
            elif request['ticket_id'] is not None or request['expected_version'] is not None:raise Fault(422,'unexpected_ticket_fields')
            if action!='ask' and p['role']!='agent':raise Fault(403,'agent_required')
            id_,old=self.store.begin(p,request['request_key'],digest(request))
            if old:
                self.revalidate_result(old,p);return old
            trace=[];payload=None
            def stage(name):
                if len(trace)>=request['max_steps']:raise Fault(429,'step_budget_exhausted')
                trace.append(dict(stage=name,status='ok'))
            try:
                stage('authorize')
                if action in ['note','resolve']:
                    ticket=self.store.ticket(request['ticket_id'],p)
                    if ticket['version']!=request['expected_version']:raise Fault(409,'stale_ticket_version')
                    if ticket['status']!='open':raise Fault(409,'ticket_closed')
                stage('retrieve')
                if self.down:raise Fault(503,'retrieval_unavailable')
                evidence,stats=retrieve(self.docs,p,request['service'],request['message'],self.manifest['min_terms'],600,self.revoked)
                stage('validate')
                citations=copy.deepcopy(evidence[:1])
                if self.corrupt and citations:citations[0]['quote']+=' unsupported claim'
                if citations and not validate(citations,evidence,self.docs,p,request['service'],self.revoked):raise Fault(503,'citation_validation_failed')
                if not citations:result=dict(state='abstained',code='insufficient_evidence',answer='',citations=[],http_status=200,stats=stats)
                else:
                    result=dict(state='answered',code='ok',answer=citations[0]['quote'],citations=citations,http_status=200,stats=stats)
                    if action!='ask':
                        stage('propose')
                        payload=dict(operation=action,ticket_id=request['ticket_id'],expected_version=request['expected_version'],title=request['message'][:80],body=request['message'],service=request['service'],citations=citations,release=self.release)
                        result['state']='proposed';result['code']='review_required'
            except Fault as error:
                result=dict(state='failed',code=error.code,answer='',citations=[],http_status=error.status,stats={})
                if trace and error.code!='step_budget_exhausted':trace[-1]['status']='failed'
            result.update(release=self.release,service=request['service'],trace=trace,steps=len(trace),proposal_id=None)
            return self.store.finish(id_,result,payload)

    def revalidate_result(self,result,p):
        cites=result['citations']
        if cites and not validate(cites,cites,self.docs,p,result['service'],self.revoked):raise Fault(409,'evidence_no_longer_allowed')

    def read_run(self,token,id_):
        with self.store.lock:
            p=self.principal(token);self.integrity();result=self.store.get_run(id_,p);self.revalidate_result(result,p);return result

    def read_proposal(self,token,id_):
        with self.store.lock:
            p=self.principal(token);self.integrity();row=self.store.proposal(id_,p,review=p['role']=='reviewer')
            payload=row['payload'];cites=payload['citations']
            if not validate(cites,cites,self.docs,p,payload['service'],self.revoked):raise Fault(409,'evidence_no_longer_allowed')
            return {k:row[k] for k in ['id','payload','payload_hash','state','expires']}

    def review(self,token,id_):
        with self.store.lock:
            p=self.principal(token)
            if p['role']!='reviewer':raise Fault(403,'reviewer_required')
            self.read_proposal(token,id_);row=self.store.proposal(id_,p,review=True)
            if row['payload']['release']!=self.release:raise Fault(409,'release_changed')
            self.validate_payload(row['payload'],self.identities_by_id(row['actor']))
            return self.store.approve(id_,p,self.clock())

    def identities_by_id(self,id_):
        found=[p for p in self.identities.values() if p['id']==id_ and p['active']]
        if len(found)!=1:raise Fault(401,'proposal_owner_inactive')
        return found[0]

    def validate_payload(self,payload,p):
        self.authenticate(p);self.integrity()
        if p['role']!='agent':raise Fault(403,'agent_required')
        if set(payload)!=set(['operation','ticket_id','expected_version','title','body','service','citations','release']) or payload['operation'] not in ['create','note','resolve']:raise Fault(422,'invalid_proposal')
        if not validate(payload['citations'],payload['citations'],self.docs,p,payload['service'],self.revoked):raise Fault(409,'evidence_no_longer_allowed')

    def commit(self,token,id_,key,fault=None):
        with self.store.lock:
            if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',key):raise Fault(422,'invalid_request_key')
            p=self.principal(token)
            return self.store.execute(id_,p,key,self.clock(),self.release,self.validate_payload,fault)


def request(message='timeout 504',action='ask',key='example',service='gateway',ticket_id=None,expected_version=None,max_steps=8):
    return dict(request_key=key,action=action,service=service,message=message,ticket_id=ticket_id,expected_version=expected_version,max_steps=max_steps)
