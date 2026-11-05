"""Actual asyncio fan-out of independent reads; scaled sleep simulates I/O, not a benchmark."""
import asyncio
import time
from backend import Backend,ToolError
from contracts import call,validate_output,pair_results


async def read_fanout(backend,subject='ops_blue',service='checkout',optional_health=False,timeout_s=.5):
    calls=[call('p-kb','lookup_runbook',service=service),call('p-health','get_health',service=service)]
    completion=[]
    async def one(c):
        # 80/120 modeled ms -> 8/12 ms asyncio sleep, no network.
        await asyncio.sleep(backend.latency[c['name']]/10000)
        try:
            data,_=backend.execute(subject,c)
            if not validate_output(c['name'],data):raise ToolError('invalid_output')
            result=dict(call_id=c['id'],name=c['name'],status='ok',data=data,error=None)
        except ToolError as exc:
            if not (optional_health and c['name']=='get_health' and exc.code in {'unavailable','timeout_before'}):raise
            result=dict(call_id=c['id'],name=c['name'],status='error',data=None,error=exc.code)
        completion.append(c['id']);return result
    started=time.perf_counter()
    async with asyncio.timeout(timeout_s):
        async with asyncio.TaskGroup() as group:
            tasks=[group.create_task(one(c)) for c in calls]
    results=pair_results(calls,[t.result() for t in tasks])
    return dict(results=results,completion_order=completion,wall_ms=(time.perf_counter()-started)*1000,
                note='Actual asyncio with scaled synthetic sleep, not network/service latency')


def hand_schedule():
    return dict(ticket_ms=40,runbook_ms=80,health_ms=120,draft_ms=50,write_ms=60,
                sequential_ms=40+80+120+50+60,parallel_critical_path_ms=40+max(80,120)+50+60,
                total_tool_work_ms=40+80+120+60,backoff_ms=[50,100,200],
                chain_success_if_independent=.98*.97*.99,retry3_if_iid=1-(1-.8)**3)

if __name__=='__main__':
    import json
    print(json.dumps(asyncio.run(read_fanout(Backend())),ensure_ascii=False,indent=2))
