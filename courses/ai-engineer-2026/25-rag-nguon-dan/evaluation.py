"""Closed reference evaluator. Annotations are visible only here, never to generator."""
from data import claim_catalog


def covered_facts(span,facts):
    return {fid for fid,f in facts.items() if any(
        s['source_id']==span['source_id'] and s['version']==span['version']
        and span['source_start']<=s['start'] and s['end']<=span['source_end'] for s in f['spans'])}


def facts_in_chunks(chunks,facts):
    result=set()
    for c in chunks:
        result |= covered_facts(dict(source_id=c['source_id'],version=c['version'],
                                     source_start=c['start'],source_end=c['end']),facts)
    return result


def coverage(query,chunks,facts):
    required=set(query['required'])
    if not required:return None
    found=required&facts_in_chunks(chunks,facts)
    return dict(required_recall=len(found)/len(required),complete=float(found==required))


def evaluate_candidate(query,validation,facts):
    # Parse/schema failures remain failures in the task denominator.
    if not validation['schema_valid']:
        return dict(task_success=0.,claim_count=0,citation_count=0,attached_claims=0,
                    supported_claims=0,supporting_citations=0,unknown_claims=0,rows=[],
                    required_fact_recall=None,all_required=False)
    catalog=claim_catalog(facts);obj=validation['candidate'];rows=[]
    recovered=set();supported_count=0;precise=0;unknown=0
    for i,claim in enumerate(obj['claims']):
        ann=catalog.get(claim['text'])
        inspected=[r for r in validation['citations'] if r['claim_index']==i]
        sets=[covered_facts(r,facts) if r['valid'] else set() for r in inspected]
        union=set().union(*sets) if sets else set()
        supported=bool(ann and any(required<=union for required in ann['support_sets']))
        useful=set().union(*ann['support_sets']) if ann and ann['support_sets'] else set()
        edge_flags=[bool(supported and s&useful) for s in sets]
        if supported:
            recovered |= ann['entails'];supported_count+=1
        precise+=sum(edge_flags);unknown+=int(ann is None)
        rows.append(dict(text=claim['text'],known=ann is not None,supported=supported,
                         supporting_citations=edge_flags,covered_fact_ids=sorted(union)))
    required=set(query['required']);all_required=required<=recovered
    status_ok=obj['status']==query['expected_status']
    all_supported=supported_count==len(obj['claims'])
    success=bool(validation['accepted'] and status_ok and all_required and all_supported)
    return dict(task_success=float(success),claim_count=len(obj['claims']),
                citation_count=sum(len(c['citations']) for c in obj['claims']),
                attached_claims=sum(bool(c['citations']) for c in obj['claims']),
                supported_claims=supported_count,supporting_citations=precise,unknown_claims=unknown,
                required_fact_recall=len(required&recovered)/len(required) if required else None,
                all_required=all_required,rows=rows)


def aggregate(rows):
    # Micro citation/claim metrics + macro task success; zero denominator => None.
    counts={k:sum(r['semantic'][k] for r in rows) for k in
            ['claim_count','citation_count','attached_claims','supported_claims','supporting_citations','unknown_claims']}
    def ratio(a,b):return a/b if b else None
    return dict(cases=len(rows),parsed=sum(r['validation']['parsed'] for r in rows),
                schema_valid=sum(r['validation']['schema_valid'] for r in rows),
                mechanically_accepted=sum(r['validation']['accepted'] for r in rows),
                task_successes=sum(r['semantic']['task_success'] for r in rows),
                task_success_rate=sum(r['semantic']['task_success'] for r in rows)/len(rows),
                citation_attachment=ratio(counts['attached_claims'],counts['claim_count']),
                grounded_claim_rate=ratio(counts['supported_claims'],counts['claim_count']),
                citation_precision=ratio(counts['supporting_citations'],counts['citation_count']),**counts)
