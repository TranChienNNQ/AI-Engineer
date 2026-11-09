import asyncio,httpx
from bridge53 import Fault
from contracts53 import Receipt
from target53 import build_app
class UnknownOutcome(Exception):pass
class Connector:
 def __init__(self,path,token='demo-agent-a',now=100,fault=None):self.path=path;self.token=token;self.now=now;self.fault=fault
 def request(self,method,path,payload=None):
  async def send():
   app=build_app(self.path,self.now,self.fault)
   async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://synthetic-ticket',headers={'Authorization':'Bearer '+self.token},timeout=1.) as c:
    return await c.request(method,path,json=payload)
  # Fault before_send/after_commit models transport ambiguity; no real TCP timeout measurement.
  if method=='POST' and self.fault=='before_send':raise UnknownOutcome('simulated_before_send')
  try:response=asyncio.run(send())
  except (httpx.TransportError,TimeoutError):raise UnknownOutcome('transport_unknown') from None
  if method=='POST' and self.fault=='after_commit' and response.status_code==200:raise UnknownOutcome('simulated_lost_ack')
  if response.status_code>=500:raise UnknownOutcome('server_error_unknown')
  if response.status_code!=200:raise Fault(response.status_code,response.json().get('error','target_rejected'))
  return response.json()
 def ticket(self,id_):return self.request('GET',f'/v1/tickets/{id_}')
 def apply(self,envelope):
  data=self.request('POST','/v1/notes',envelope)
  if self.fault=='bad_receipt':data=data|dict(ticket_id=2)
  return Receipt.model_validate(data).model_dump()
 def status(self,op_id):return self.request('GET','/v1/operations/'+op_id)
