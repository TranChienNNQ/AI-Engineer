"""Regression on definitions, bounded generator, ASGI behavior and accounting; no speed threshold."""
import asyncio
import json
from decimal import Decimal
import httpx
from measurements import quantile,distribution,summarize,streaming_example
from simulation import simulate,scenarios
from costs import attempt_cost,account,ledger,self_hosted_example
from serving import Service,create_app,valid_response
from loadgen import Config,run,PAYLOAD


async def evaluate():
    rows=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        rows.append(dict(name=name,passed=True))
    def rejects(fn):
        try:fn()
        except (ValueError,TypeError):return True
        return False
    values=[10,20,30,40,500]
    check('nearest_rank_p95',quantile(values,.95)==500)
    check('mean_is_not_tail',distribution(values)['mean']==120)
    check('empty_percentile_none',quantile([],.95) is None and distribution([])['mean'] is None)
    check('reject_nan_latency',rejects(lambda:quantile([float('nan')],.95)))
    check('reject_bool_latency',rejects(lambda:quantile([True],.95)))
    check('reject_invalid_quantile',rejects(lambda:quantile([1],0)))
    s=scenarios();burst=s['bounded_burst'];closed=s['closed_one'];opened=s['open_same_demand']
    check('burst_capacity_counts',burst['summary']['statuses']=={'200':4,'503':4})
    check('burst_queue_bound',burst['peak_queue']==2)
    check('burst_goodput_differs',burst['summary']['successful']==4 and burst['summary']['good']==2)
    check('burst_fifo_wait',[r['start_ms'] for r in burst['rows'][:4]]==[0,0,20,20])
    check('rejection_not_success_latency',burst['summary']['latency_success_ms']['n']==4 and burst['summary']['latency_errors_ms']['n']==4)
    check('closed_load_waits',closed['summary']['latency_success_ms']['p95']==100)
    check('open_backlog_visible',opened['summary']['latency_success_ms']['p95']==820)
    check('simulation_utilization_bound',all(0<=v['worker_utilization']<=1 for v in s['sweep'].values()))
    check('simulation_conservation',all(v['summary']['successful']+v['summary']['errors']==40 for v in s['sweep'].values()))
    check('same_time_finish_before_arrival',simulate([0,20],workers=1,queue_cap=0)['summary']['successful']==2)
    check('empty_simulation',simulate([])['summary']['success_fraction_of_planned'] is None)
    check('invalid_simulation_clock',rejects(lambda:simulate([1,0])) and rejects(lambda:simulate([0],service_ms=float('nan'))))
    delayed=[dict(id=0,planned_ms=0,sent_ms=100,done_ms=120,status=200),dict(id=1,planned_ms=10,sent_ms=None,done_ms=100,status='generator_drop')]
    d=summarize(delayed,200,40)
    check('scheduled_latency_includes_generator',d['latency_success_ms']['mean']==20 and d['scheduled_latency_success_ms']['mean']==120)
    check('dropped_in_planned_denominator',d['planned']==2 and d['sent']==1 and d['attainment_of_planned']==0)
    check('ttft_and_tpot',streaming_example()['ttft_ms']==120 and streaming_example()['tpot_using_response_end_ms']==40)
    check('tpot_boundary_difference',streaming_example()['finish_overhead_ms']==20 and streaming_example()['tpot_to_last_token_ms']!=40)
    costs=[attempt_cost(r['usage']) for r in ledger()]
    check('cached_is_subset',costs==[Decimal('.0028'),Decimal('.00285'),Decimal('.0049')])
    result=account(ledger())
    check('failed_attempt_charged',result['total_cost_usd']=='0.01055')
    check('task_and_attempt_differ',result['tasks']==2 and result['attempts']==3)
    check('qualified_denominator',result['cost_per_qualified_task_usd']=='0.01055' and result['cost_per_successful_task_usd']=='0.005275')
    check('unknown_usage_not_free',self_hosted_example()['missing_usage']['total_cost_usd'] is None)
    check('empty_ledger_no_divide_zero',account([])['cost_per_qualified_task_usd'] is None)
    check('duplicate_attempt_rejected',rejects(lambda:account(ledger()+ledger()[:1])))
    check('cached_over_total_rejected',rejects(lambda:attempt_cost(dict(input=10,cached_input=11,output=0))))
    check('negative_usage_rejected',rejects(lambda:attempt_cost(dict(input=-1,cached_input=0,output=0))))
    check('boolean_usage_rejected',rejects(lambda:attempt_cost(dict(input=True,cached_input=0,output=0))))
    check('allocated_idle_hour_in_cost',self_hosted_example()['usd_per_request']=='0.00005')
    check('config_bounds',rejects(lambda:Config(count=100000).validate()) and rejects(lambda:Config(rate=float('nan')).validate()))
    service=Service(workers=1,queue_cap=1,delay=0,gate=asyncio.Event())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)),base_url='http://lab') as client:
        first=asyncio.create_task(client.post('/v1/predict',json=PAYLOAD));await asyncio.wait_for(service.entered.wait(),1)
        second=asyncio.create_task(client.post('/v1/predict',json=PAYLOAD))
        async with asyncio.timeout(1):
            while service.outstanding!=2:await asyncio.sleep(0)
        third=await client.post('/v1/predict',json=PAYLOAD)
        check('service_overload_503',third.status_code==503)
        check('service_counts_bounded',service.active==1 and service.outstanding==2 and service.peak_waiting==1)
        service.gate.set();a,b=await asyncio.gather(first,second)
        check('accepted_finish_200',[a.status_code,b.status_code]==[200,200])
        check('numeric_response_verified',valid_response(a.json(),PAYLOAD))
        broken=a.json()|{'priority':0};check('http200_not_enough',not valid_response(broken,PAYLOAD))
        check('service_drained',service.outstanding==service.active==0)
        check('validation_422',(await client.post('/v1/predict',json={'latency_ratio':True,'error_ratio':0.0})).status_code==422)
    # Deterministic blocked fake transport: generator emits 4 planned arrivals, cap1 means 3 drops.
    hold=asyncio.Event();arrived=asyncio.Event()
    async def blocked(request):
        arrived.set();await hold.wait();return httpx.Response(503,json={'code':'test'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(blocked),base_url='http://lab') as client:
        emitted=asyncio.Event()
        task=asyncio.create_task(run(client,Config(rate=500,count=4,max_inflight=1,max_lag_ms=1000),emitted))
        await asyncio.wait_for(arrived.wait(),1);await asyncio.wait_for(emitted.wait(),1);hold.set();r=await task
    check('generator_cap_visible',r['summary']['generator_dropped']==3 and r['summary']['sent']==1)
    check('generator_drop_not_server_error',r['summary']['errors']==1 and r['summary']['planned']==4)
    async def slow(request):await asyncio.sleep(.1);return httpx.Response(503)
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow),base_url='http://lab') as client:
        r=await run(client,Config(mode='closed',count=1,request_timeout_ms=1))
    check('total_timeout_recorded',r['rows'][0]['status']=='total_timeout' and r['summary']['successful']==0)
    check('timeout_not_imputed_latency',r['summary']['latency_success_ms']['n']==0 and r['summary']['latency_errors_ms']['n']==1)
    return dict(kind='Definition/accounting/ASGI/generator regression, not a performance or task-quality benchmark',checks=len(rows),passed=len(rows),rows=rows)


def validate_tcp(tcp):
    checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(dict(name=name,passed=True))
    check('four_profiles_per_repeat',len(tcp['runs'])==tcp['repeats']*4)
    for r in tcp['runs']:
        s=r['summary'];server=r['server']
        check(r['profile']+f"-r{r['repetition']}-client_conservation",s['planned']==s['sent']+s['generator_dropped']==len(r['rows']))
        check(r['profile']+f"-r{r['repetition']}-terminal_conservation",s['sent']==s['successful']+s['errors'])
        check(r['profile']+f"-r{r['repetition']}-goodput_subset",0<=s['good']<=s['successful'] and s['goodput_rps']<=s['success_rps']<=s['attempt_rps'])
        check(r['profile']+f"-r{r['repetition']}-server_bounds",server['peak_active']<=2 and server['peak_waiting']<=4 and server['outstanding_after_drain']==0)
        check(r['profile']+f"-r{r['repetition']}-time_order",all(row['done_ms']>=row['planned_ms'] and (row['sent_ms'] is None or row['planned_ms']<=row['sent_ms']<=row['done_ms']) for row in r['rows']))
        check(r['profile']+f"-r{r['repetition']}-quantile_population",s['latency_success_ms']['n']==s['successful'] and s['dispatch_lag_ms']['n']==s['sent'])
    return dict(kind='Structural TCP run checks; no throughput/p95 acceptance threshold',checks=len(checks),passed=len(checks),rows=checks)
