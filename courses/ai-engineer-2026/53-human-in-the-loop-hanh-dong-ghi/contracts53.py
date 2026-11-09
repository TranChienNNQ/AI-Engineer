import hmac,hashlib
from typing import Literal
from pydantic import Field
from bridge53 import Strict,digest,canonical
VERSION='53.1'
# Deliberately public teaching key: verifies binding, NOT a production credential boundary.
DEMO_KEY=b'lesson53-public-demo-key-never-use-for-real-services'
class Draft(Strict):
 version:int=Field(ge=1,le=5)
 operation:Literal['append_internal_note']
 ticket_id:int=Field(ge=1)
 expected_version:int=Field(ge=1)
 service:Literal['gateway']
 body:str=Field(min_length=1,max_length=1200)
 source_binding:str=Field(pattern='^[a-f0-9]{64}$')
 evidence_hash:str=Field(pattern='^[a-f0-9]{64}$')
 policy_release:str=Field(pattern='^[a-f0-9]{64}$')
 expires_at:int
class Receipt(Strict):
 op_id:str
 ticket_id:int
 version:int
 draft_hash:str
 state:Literal['applied']
class Permit(Strict):
 op_id:str
 draft_hash:str
 tenant:str
 actor:str
 reviewer:str
 expires_at:int
class Envelope(Strict):
 op_id:str=Field(pattern='^[a-zA-Z0-9_-]{1,60}$')
 draft:Draft
 permit:Permit
 signature:str=Field(pattern='^[a-f0-9]{64}$')

def sign(permit):return hmac.new(DEMO_KEY,canonical(permit).encode('utf-8'),hashlib.sha256).hexdigest()
def signed(permit):return dict(permit=permit,signature=sign(permit))
def signature_valid(permit,signature):return hmac.compare_digest(sign(permit),signature)
