"""Explicit sample populations and nearest-rank percentiles for the lesson."""
import math
from fractions import Fraction


def quantile(values,q):
    if not 0<q<=1:raise ValueError('q must be in (0,1]')
    values=sorted(values)
    if any(type(v) not in {int,float,Fraction} or not math.isfinite(v) or v<0 for v in values):raise ValueError('invalid observation')
    return values[math.ceil(q*len(values))-1] if values else None


def distribution(values):
    values=list(values)
    return dict(n=len(values),mean=sum(values)/len(values) if values else None,
                p50=quantile(values,.5),p95=quantile(values,.95),p99=quantile(values,.99),maximum=max(values) if values else None)


def summarize(rows,window_ms,slo_ms):
    if window_ms<=0 or slo_ms<0:raise ValueError('invalid window or SLO')
    sent=[r for r in rows if r['sent_ms'] is not None]
    success=[r for r in sent if r['status']==200]
    errors=[r for r in sent if r['status']!=200]
    good=[r for r in success if r['done_ms']-r['planned_ms']<=slo_ms]
    statuses={str(s):sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows},key=str)}
    return dict(planned=len(rows),sent=len(sent),generator_dropped=len(rows)-len(sent),successful=len(success),
                errors=len(errors),good=len(good),statuses=statuses,
                window_ms=window_ms,attempt_rps=len(sent)/(window_ms/1000),success_rps=len(success)/(window_ms/1000),goodput_rps=len(good)/(window_ms/1000),
                success_fraction_of_planned=len(success)/len(rows) if rows else None,
                attainment_of_planned=len(good)/len(rows) if rows else None,
                latency_success_ms=distribution(r['done_ms']-r['sent_ms'] for r in success),
                latency_errors_ms=distribution(r['done_ms']-r['sent_ms'] for r in errors),
                scheduled_latency_success_ms=distribution(r['done_ms']-r['planned_ms'] for r in success),
                dispatch_lag_ms=distribution(r['sent_ms']-r['planned_ms'] for r in sent))


def streaming_example():
    # Timestamps of individual tokens for one hypothetical request, not SSE chunks.
    sent=0;tokens=[120,150,190,220];done=240
    return dict(kind='hypothetical token timestamps, no streaming inference executed',
                token_times_ms=tokens,ttft_ms=tokens[0]-sent,e2e_ms=done-sent,
                tpot_to_last_token_ms=(tokens[-1]-tokens[0])/(len(tokens)-1),
                itl_ms=[b-a for a,b in zip(tokens,tokens[1:])],
                finish_overhead_ms=done-tokens[-1],
                tpot_using_response_end_ms=(done-tokens[0])/(len(tokens)-1))
