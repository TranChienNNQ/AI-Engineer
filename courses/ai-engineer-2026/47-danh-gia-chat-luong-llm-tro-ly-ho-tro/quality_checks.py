import copy,json,subprocess,sys,tempfile
from pathlib import Path
from bridge46 import digest,documents
from dataset import cases,validate_split,CONFIGS
from capture import Record
from scoring import content,join_ratings,summarize,gate,select_dev,wilson,agreement,swap_result,paired_cluster_bootstrap,load_jsonl
from runner import run,analyze,validate_records


def check(directory,report=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if report is None:report=run(directory/'capture')
    path=directory/'capture' if (directory/'capture/records.jsonl').exists() else directory.parent
    records=load_jsonl(path/'records.jsonl');ratings=load_jsonl(path/'author-ratings.jsonl');joined=join_ratings(records,ratings)
    names=[]
    def ok(name,condition):
        if not condition:raise AssertionError(name)
        names.append(name)
    def rejects(name,fn):
        try:fn()
        except (ValueError,TypeError):names.append(name)
        else:raise AssertionError(name)
    ok('16_cases_8_groups_8_dev_8_test',validate_split(cases())|{}==dict(cases=16,groups=8,dev=8,test=8,snapshot=digest(cases())))
    leaked=copy.deepcopy(cases());leaked[1]['split']='test'
    rejects('group_cross_split',lambda:validate_split(leaked))
    duplicate=copy.deepcopy(cases());duplicate[1]['query']='  '+duplicate[0]['query'].upper()+'  '
    rejects('normalized_duplicate',lambda:validate_split(duplicate))
    rejects('duplicate_case_id',lambda:validate_split(cases()+[cases()[0]]))
    ok('96_paired_outputs',len(records)==96 and len(validate_records(records))==96)
    rejects('missing_output_not_dropped',lambda:validate_records(records[:-1]))
    rejects('duplicate_sample_not_weighted_twice',lambda:validate_records(records+[records[0]]))
    corrupt=copy.deepcopy(records[0]);corrupt['answer']+=' changed'
    rejects('content_hash_binding',lambda:Record.model_validate(corrupt))
    corrupt=copy.deepcopy(records[0]);corrupt['code']='changed'
    rejects('full_record_hash_binding',lambda:Record.model_validate(corrupt))
    r=copy.deepcopy(ratings[0]);r['content_hash']='0'*64
    rejects('stale_rating',lambda:join_ratings(records,[r]))
    r=copy.deepcopy(ratings[0]);r['rubric']='old'
    rejects('rubric_version',lambda:join_ratings(records,[r]))
    rejects('duplicate_rating',lambda:join_ratings(records,[ratings[0],ratings[0]]))
    r=copy.deepcopy(ratings[0]);r['item_id']='unknown'
    rejects('unknown_rating',lambda:join_ratings(records,[r]))
    for value in [True,2.0,'2',-1,3,None]:
        r=copy.deepcopy(ratings[0]);r['support']=value
        rejects('strict_support_'+repr(value),lambda r=r:join_ratings(records,[r]))
    r=copy.deepcopy(ratings[0]);r['critical']=1
    rejects('strict_critical',lambda:join_ratings(records,[r]))
    r=copy.deepcopy(ratings[0]);r['reason']=''
    rejects('rating_reason_required',lambda:join_ratings(records,[r]))
    r=copy.deepcopy(ratings[0]);r['extra']='authority'
    rejects('rating_extra_field',lambda:join_ratings(records,[r]))
    r=copy.deepcopy(next(a for a in ratings if a['draft'] is not None));r['draft']=None
    rejects('draft_applicability',lambda:join_ratings(records,[r]))
    unrated=summarize(records,{})
    ok('missing_ratings_remain_unknown',unrated['unrated']==unrated['served'] and unrated['rating_coverage']==0)
    ok('missing_ratings_gate_fails','missing_ratings' in gate(unrated)['reasons'])
    ok('not_select_missing_ratings',select_dev({'missing':unrated}) is None)
    ok('no_candidate_promoted',report['selected_on_dev'] is None and report['selected_test'] is None)
    ok('critical_gate_overrides_score',all('critical' in x['reasons'] for x in report['dev_gate'].values()))
    for cfg in CONFIGS:
        for split in ['dev','test']:
            s=report['summary'][cfg][split]
            ok(f'{cfg}_{split}_denominator_24',s['rows']==24)
            ok(f'{cfg}_{split}_all_served_rated',s['all_eligible_rated'] and s['unrated']==0)
    for cfg in CONFIGS:
        s=report['summary'][cfg]['test']
        ok(cfg+'_incomplete_kept_in_denominator',sum(r['code']=='provider_incomplete' for r in records if r['config']==cfg and r['split']=='test')==1 and s['rows']==24)
    ok('conditional_differs_end_to_end',report['summary']['quote_guard']['dev']['quality_conditional']!=report['summary']['quote_guard']['dev']['task_success'])
    ok('citation_judge_false_accepts',report['judge_audit']['vs_author']['fp']>0 and report['judge_audit']['vs_author']['fn']==0)
    ok('kappa_degenerate_none',agreement([True,True],[True,True])['kappa'] is None)
    ok('agreement_known',agreement([True,True,False,False],[True,False,True,False])==dict(tp=1,fp=1,fn=1,tn=1,agreement=.5,kappa=0.,bad_pass_rate=.5))
    ok('pair_swap_mapping',swap_result('A','B')['verdict']=='A')
    ok('pair_position_bias_review',swap_result('A','A')['verdict']=='needs_review')
    ok('pair_consistent_tie',swap_result('tie','tie')['consistent'])
    rejects('pair_invalid_label',lambda:swap_result('first','A'))
    ok('zero_failures_nonzero_upper',.11<wilson(0,30)[1]<.12)
    ok('wilson_symmetry',abs(wilson(0,30)[1]-(1-wilson(30,30)[0]))<1e-12)
    rejects('wilson_empty_denominator',lambda:wilson(0,0))
    rejects('wilson_bad_count',lambda:wilson(31,30))
    rejects('wilson_boolean',lambda:wilson(True,30))
    left={'a':0.,'b':1.};right={'a':.5,'b':1.}
    ci=paired_cluster_bootstrap(left,right)
    ok('paired_delta_known',ci['delta']==.25 and ci['low']==0 and ci['high']==.5)
    ok('bootstrap_reproducible',ci==paired_cluster_bootstrap(left,right))
    rejects('bootstrap_misaligned',lambda:paired_cluster_bootstrap(left,{'c':1.}))
    ok('test_resampled_by_4_groups',report['test_pair_diagnostic']['groups']==4)
    ok('no_ticket_effects',all(r['ticket_count']==2 and r['receipt_count']==0 for r in records))
    ok('permission_and_unknown_bypass',all(r['transport_calls']==0 and r['state']=='abstained' for r in records if r['expected']=='abstain'))
    ok('critical_drafts_not_approved',any(r['variant']=='invented_draft' and r['served'] and joined[r['item_id']]['critical'] for r in records))
    docs={(d['id'],d['version']):d for d in documents()};valid=True
    for r in records:
        for cite in r['citations']:
            d=docs[(cite['doc_id'],cite['version'])]
            valid &= d['text'][cite['start']:cite['end']]==cite['quote'] and digest(d['text'])==cite['content_hash']
    ok('citation_offsets_and_hashes',valid)
    queue=load_jsonl(path/'review-queue.jsonl')
    ok('blind_queue_no_config_variant_truth',all(not ({'config','repeat','variant','expected','case_id'} & q.keys()) for q in queue))
    ok('blind_queue_all_served',len(queue)==sum(r['served'] for r in records))
    p=directory/'bad.jsonl';p.write_text('{"x":1,"x":2}\n')
    rejects('duplicate_json_keys',lambda:load_jsonl(p));p.write_text('{"x":NaN}\n')
    rejects('nonfinite_json',lambda:load_jsonl(p))
    # Selection is frozen by a function with dev-only input. Test diagnostic cannot select.
    changed=copy.deepcopy(report['summary']);changed['quote_guard']['test']['task_success']=1.
    ok('selection_ignores_test_scores',select_dev({c:s['dev'] for c,s in changed.items()})==report['selected_on_dev'])
    base=Path(__file__).resolve().parent.parent/'46-tich-hop-llm-tro-ly-ho-tro-ky-thuat'
    done=subprocess.run([sys.executable,'lab.py','--mode','check'],cwd=base,capture_output=True,text=True,timeout=120)
    if done.returncode:raise AssertionError(done.stderr)
    old=json.loads(done.stdout)
    if old['passed']!=78 or old['base45_checks']!=75:raise AssertionError('base46 regression failed')
    return dict(checks=len(names),passed=len(names),names=names,base46_checks=78,base45_checks=75,total_predicates=len(names)+153)
