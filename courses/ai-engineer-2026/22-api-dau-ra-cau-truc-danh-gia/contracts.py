"""Versioned request/response and JSON Schema contracts; no provider-specific SDK."""
import json
from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from jsonschema import Draft202012Validator

LineID = Annotated[str, Field(pattern=r'^L[1-9][0-9]*$')]

class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')

class Line(StrictModel):
    id: LineID
    text: str = Field(min_length=1, max_length=500)

class RouteRequest(StrictModel):
    schema_version: Literal['1']
    ticket_id: str = Field(min_length=1, max_length=40)
    lines: list[Line] = Field(min_length=1, max_length=8)

    @model_validator(mode='after')
    def unique_ids(self):
        if len({line.id for line in self.lines}) != len(self.lines):
            raise ValueError('Duplicate line IDs')
        return self

BRANCHES = [
    {'properties': {'decision': {'const': 'route'}, 'team': {'enum': ['AUTH','DATA','NET']},
                    'evidence_ids': {'minItems': 1}}},
    {'properties': {'decision': {'const': 'abstain'}, 'team': {'type': 'null'},
                    'evidence_ids': {'maxItems': 0}}}
]

class Proposal(StrictModel):
    model_config = ConfigDict(strict=True, extra='forbid', json_schema_extra={'oneOf': BRANCHES})
    schema_version: Literal['1']
    decision: Literal['route','abstain']
    team: Literal['AUTH','DATA','NET'] | None
    evidence_ids: list[LineID] = Field(max_length=8, json_schema_extra={'uniqueItems': True})

    @model_validator(mode='after')
    def check_relationships(self):
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError('Repeated evidence ID')
        if self.decision == 'route' and (self.team is None or not self.evidence_ids):
            raise ValueError('Route requires team and evidence')
        if self.decision == 'abstain' and (self.team is not None or self.evidence_ids):
            raise ValueError('Abstain requires null team and empty evidence')
        return self

OUTPUT_SCHEMA = Proposal.model_json_schema()
OUTPUT_SCHEMA['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
Draft202012Validator.check_schema(OUTPUT_SCHEMA)
OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)

class Metrics(StrictModel):
    attempts: int
    elapsed_ms: float
    known_input_tokens: int
    known_output_tokens: int
    usage_complete: bool

class Success(StrictModel):
    request_id: str
    provider: str
    prompt_version: Literal['route-v1']
    result: Proposal
    metrics: Metrics

class ErrorBody(StrictModel):
    request_id: str
    error: str
    attempts: int

class ContractError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def parse_json(raw):
    if not isinstance(raw,str):
        raise ContractError('invalid_json')
    try:
        byte_count = len(raw.encode('utf-8'))
    except UnicodeError:
        raise ContractError('invalid_json') from None
    if byte_count > 16000:
        raise ContractError('invalid_json')
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result: raise ValueError('Duplicate key')
            result[key] = value
        return result
    def constant(value): raise ValueError('Nonstandard JSON number')
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError):
        raise ContractError('invalid_json') from None


def validate_candidate(raw, request):
    obj = parse_json(raw)
    if list(OUTPUT_VALIDATOR.iter_errors(obj)):
        raise ContractError('invalid_schema')
    proposal = Proposal.model_validate(obj)
    available = {line.id for line in request.lines}
    if not set(proposal.evidence_ids).issubset(available):
        raise ContractError('unknown_evidence')
    # IDs existing does not prove they support the proposed team. Task evaluation is separate.
    return proposal


def export_schema(path):
    Path(path).write_text(json.dumps(OUTPUT_SCHEMA,ensure_ascii=False,indent=2)+'\n')
