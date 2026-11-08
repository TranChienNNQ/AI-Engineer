"""Regressions of telemetry structure and quality joins; not model accuracy benchmark."""
from dataclasses import replace
import asyncio
import json
import math
from data import dataset,Prediction,Feedback
from telemetry import Telemetry,Histogram,extract,inject
from serving import Runtime,client_for,API,MODEL
from demos import trace_scenarios,hand_demo,PAYLOAD,capture
from quality import quality_demo,snapshot,joined_labels,score_metrics,distribution_compare
from opentelemetry import trace


async def evaluate(tracing=None,quality=None):
    t=tracing or await trace_scenarios();q=quality or quality_demo();rows=[]
    def record(name,predicate):
        if not predicate:raise AssertionError(name)
        rows.append(dict(name=name,passed=True))
    normal=t['normal'];spans=normal['spans'];by_id={s['span_id']:s for s in spans}
    record('normal_requests_succeed',normal['statuses']==[200,200] and len(normal['predictions'])==2)
    record('seven_spans_per_prediction',normal['trace_count']==2 and normal['span_count']==14)
    record('incoming_w3c_ids_preserved',{s['trace_id'] for s in spans}=={'1'*32,'3'*32})
    record('unique_span_ids',len(by_id)==14)
    record('sdk_span_timestamps',all(s['end_ns']>=s['start_ns'] for s in spans))
    expected={'queue.wait':'POST /v1/predict','queue.publish':'POST /v1/predict','job.process':'queue.publish',
              'model.call':'job.process','POST /score':'model.call','model.forward':'POST /score'}
    record('explicit_queue_http_parent_edges',all(by_id[s['parent_id']]['name']==expected[s['name']] for s in spans if s['name'] in expected))
    record('no_context_leak_between_requests',all(s['parent_id'] not in by_id or by_id[s['parent_id']]['trace_id']==s['trace_id'] for s in spans))
    record('producer_consumer_kinds',all(s['kind']=={'queue.publish':'PRODUCER','job.process':'CONSUMER'}[s['name']] for s in spans if s['name'] in {'queue.publish','job.process'}))
    record('broken_context_splits_trace',t['broken']['trace_count']==4 and t['broken']['span_count']==14 and t['broken']['statuses']==[200,200])
    record('logs_correlate_with_spans',all(l['span_id'] in by_id and by_id[l['span_id']]['trace_id']==l['trace_id'] for l in normal['logs']))
    record('normal_no_error_status',all(s['status']=='UNSET' for s in spans))
    record('sampling_drops_traces_not_predictions',t['unsampled']['span_count']==0 and len(t['unsampled']['predictions'])==2)
    record('sampling_drops_no_request_counters',sum(x['count'] for x in t['unsampled']['metrics']['requests'])==4)
    record('unsampled_logs_still_correlated',len(t['unsampled']['logs'])==4 and all(len(x['trace_id'])==32 for x in t['unsampled']['logs']))
    record('export_failure_does_not_fail_prediction',t['exporter_failure']['statuses']==[200,200] and t['exporter_failure']['exporter']['failed_spans']==14)
    record('export_failure_visible',t['exporter_failure']['span_count']==0 and sum(x['count'] for x in t['exporter_failure']['metrics']['requests'])==4)
    bad=t['model_failure']
    record('model_failure_reaches_api',bad['statuses']==[503,503] and not bad['predictions'])
    record('error_status_on_failed_work',sum(s['status']=='ERROR' for s in bad['spans'])==10)
    record('exception_message_not_exported','SECRET32' not in json.dumps(dict(spans=bad['spans'],logs=bad['logs'])))
    record('overload_counts_all_api_attempts',t['overload']['statuses']==[200,200,503] and sum(x['count'] for x in t['overload']['metrics']['requests'] if x['service']==API)==3)
    record('overload_only_two_inferences',len([s for s in t['overload']['spans'] if s['name']=='model.forward'])==2)
    record('closed_providers_and_workers',all(x['provider_closed'] and x['worker_done'] for x in t.values()))
    # No raw body/credential/baggage/query reflected in telemetry; context header is metadata, not identity.
    runtime=Runtime()
    async with client_for(runtime) as (client,_):
        good=await client.post('/v1/predict?secret=SECRET32_QUERY',json=PAYLOAD,
                               headers={'Authorization':'Bearer SECRET32_TOKEN','baggage':'user=SECRET32_BAG'})
        invalid=await client.post('/v1/predict',json=PAYLOAD|{'raw_prompt':'SECRET32_BODY'})
        missing=await client.get('/SECRET32_PATH')
    captured=capture(runtime,[good,invalid,missing]);serialized=json.dumps(dict(spans=captured['spans'],logs=captured['logs'],metrics=captured['metrics']))
    record('telemetry_minimization_canary','SECRET32' not in serialized)
    record('validation_and_unknown_path_counted',captured['statuses']==[200,422,404] and sum(x['count'] for x in captured['metrics']['requests'] if x['service']==API)==3)
    record('route_label_is_template',all(x['route'] in {'/v1/predict','/score','unmatched'} for x in captured['metrics']['requests']))
    # Invalid all-zero W3C parent must not be accepted as a valid parent.
    context=extract({'traceparent':'00-'+'0'*32+'-'+'0'*16+'-01'})
    record('invalid_traceparent_ignored',not trace.get_current_span(context).get_span_context().is_valid)
    tel=Telemetry(export_limit=2)
    for _ in range(5):
        with tel.span(API,'bounded.test'):pass
    tel.close();record('bounded_exporter_reports_eviction',len(tel.exporter.rows)==2 and tel.exporter.evicted==3)
    hand=hand_demo();hist=hand['histogram']
    record('cumulative_histogram_hand_counts',hist['cumulative']==[2,4,4,5] and hist['count']==5 and math.isclose(hist['sum'],.6))
    record('p95_not_mean',hand['p95_nearest_rank']==.5 and math.isclose(hand['mean_seconds'],.12))
    record('critical_path_not_sum',hand['critical_path_ms']==70 and hand['sum_work_ms']==110)
    record('reference_quality',q['reference']['observed_metrics']['accuracy']==.8)
    record('late_labels_provisional',q['early']['status']=='PROVISIONAL' and q['early']['label_coverage']==.6 and q['early']['observed_metrics']['accuracy']==1)
    record('no_early_quality_alert',q['early']['alerts']==[])
    mature=q['mature'];m=mature['observed_metrics']
    record('mature_confusion_matrix',(m['tp'],m['fp'],m['fn'],m['tn'])==(1,4,0,5))
    record('mature_quality_regression',m['accuracy']==.6 and m['precision']==.2 and m['recall']==1 and math.isclose(m['brier'],.3005))
    record('slice_regression_visible',mature['slices']['mobile']['metrics']['accuracy']==.2 and mature['slices']['web']['metrics']['accuracy']==1)
    record('mature_alert_scopes',{a['scope'] for a in mature['alerts']}=={'overall','mobile'})
    record('no_drift_does_not_prove_quality',q['current_distribution']['tv']==0 and m['accuracy']<.8)
    record('drift_does_not_prove_degradation',math.isclose(q['shift_distribution']['tv'],.3) and q['shift']['observed_metrics']['accuracy']==1)
    p,f=dataset();labels,info=joined_labels(p,f+[f[0]],320)
    record('duplicate_feedback_idempotent',len(labels)==30 and info['duplicates']==1)
    conflict=replace(f[0],feedback_id='conflict',label=1-f[0].label)
    labels,info=joined_labels(p,f+[conflict],320)
    record('conflict_quarantined',len(labels)==29 and info['conflicted_predictions']==1 and info['quarantined']['label_conflict']==1)
    mismatch=replace(f[0],feedback_id='wrong-version',model_version='toy-v2')
    labels,info=joined_labels(p,[mismatch],320)
    record('version_mismatch_not_joined',not labels and info['quarantined']['unknown_prediction_or_version']==1)
    record('future_labels_not_visible',snapshot(p,f,100,110,120)['labeled']==6 and snapshot(p,f,100,110,220)['labeled']==10)
    record('cohort_boundary_half_open',snapshot(p,f,100,101,220,min_labels=1)['predictions']==1)
    record('maturity_boundary_exact',snapshot(p,f,100,110,209)['status']=='PROVISIONAL' and snapshot(p,f,100,110,210)['status']=='READY')
    sparse=snapshot(p,[x for x in f if x.prediction_id not in {'now-00','now-01','now-02','now-03'}],100,110,220)
    record('mature_missing_labels_not_perfect_quality',sparse['status']=='INSUFFICIENT_LABELS' and sparse['observed_metrics']['accuracy']==1 and sparse['alerts']==[])
    empty=snapshot(p,[],100,110,220)
    record('no_labels_not_zero_accuracy',empty['observed_metrics'] is None and empty['label_coverage']==0)
    record('quality_fixture_not_http_trace_population',len(p)==30 and len(normal['predictions'])==2)
    return dict(kind='Tracing/monitoring pipeline regression; no model-production accuracy claim',checks=len(rows),passed=len(rows),rows=rows)
