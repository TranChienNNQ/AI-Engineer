"""Hypothetical USD rate card and usage ledger. Not a provider price or invoice."""
from decimal import Decimal

D=Decimal
RATES={'uncached_input_per_million':D('2'),'cached_input_per_million':D('0.5'),'output_per_million':D('8')}


def attempt_cost(usage):
    if usage is None:return None
    if set(usage)!={'input','cached_input','output'}:raise ValueError('usage schema')
    if any(type(v) is not int or v<0 for v in usage.values()):raise ValueError('usage count')
    if usage['cached_input']>usage['input']:raise ValueError('cached exceeds total input')
    return (D(usage['input']-usage['cached_input'])*RATES['uncached_input_per_million']+
            D(usage['cached_input'])*RATES['cached_input_per_million']+
            D(usage['output'])*RATES['output_per_million'])/D(1000000)


def ledger():
    return [dict(task='A',attempt=1,status='failed',qualified=False,usage=dict(input=1000,cached_input=0,output=100)),
            dict(task='A',attempt=2,status='succeeded',qualified=True,usage=dict(input=1000,cached_input=500,output=200)),
            dict(task='B',attempt=1,status='succeeded',qualified=False,usage=dict(input=2000,cached_input=1000,output=300))]


def account(rows):
    seen=set();qualified=set();successful=set();tasks=set();known=D(0);missing=0
    for row in rows:
        key=(row['task'],row['attempt'])
        if key in seen:raise ValueError('duplicate attempt; reconcile before accounting')
        seen.add(key);tasks.add(row['task'])
        if row['status'] not in {'failed','succeeded'} or type(row['qualified']) is not bool:raise ValueError('outcome')
        if row['qualified'] and row['status']!='succeeded':raise ValueError('qualified failure')
        if row['status']=='succeeded':successful.add(row['task'])
        if row['qualified']:qualified.add(row['task'])
        cost=attempt_cost(row['usage'])
        if cost is None:missing+=1
        else:known+=cost
    complete=missing==0
    return dict(kind='hypothetical rates, provided usage; no billing API',tasks=len(tasks),attempts=len(rows),
                successful_tasks=len(successful),qualified_tasks=len(qualified),missing_usage_attempts=missing,
                known_cost_usd=str(known),total_cost_usd=str(known) if complete else None,
                cost_per_attempt_usd=str(known/len(rows)) if complete and rows else None,
                cost_per_successful_task_usd=str(known/len(successful)) if complete and successful else None,
                cost_per_qualified_task_usd=str(known/len(qualified)) if complete and qualified else None)


def self_hosted_example():
    # Fixed allocated hour, INCLUDING idle time; not milliseconds of active kernels.
    hourly=D('3');requests=60000;successes=54000;qualified=45000
    return dict(kind='hypothetical allocated full hour, one replica; no real GPU or price',
                allocated_hour_cost_usd=str(hourly),requests=requests,successes=successes,qualified=qualified,
                usd_per_request=str(hourly/requests),usd_per_success=str(hourly/successes),
                usd_per_qualified=str(hourly/qualified),
                provider_ledger=account(ledger()),missing_usage=account(ledger()+[dict(task='C',attempt=1,status='failed',qualified=False,usage=None)]))
