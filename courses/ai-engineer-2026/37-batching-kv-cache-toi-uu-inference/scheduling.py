"""Unit-tick scheduling model. No real prefill, GPU, queue network or latency prediction."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    name:str
    arrival:int
    tokens:int


def simulate(jobs,slots=2,continuous=True):
    if type(slots) is not int or slots<1 or not jobs:raise ValueError('slots/jobs')
    if len({j.name for j in jobs})!=len(jobs) or any(type(j.arrival) is not int or j.arrival<0 or type(j.tokens) is not int or j.tokens<1 for j in jobs):raise ValueError('unique jobs with nonnegative arrivals/positive tokens')
    pending=sorted(enumerate(jobs),key=lambda pair:(pair[1].arrival,pair[0]));pending=[j for _,j in pending]
    waiting=[];active=[];remaining={j.name:j.tokens for j in jobs};first={};finish={};timeline=[];tick=0
    while pending or waiting or active:
        while pending and pending[0].arrival<=tick:waiting.append(pending.pop(0))
        if continuous or not active:
            while waiting and len(active)<slots:active.append(waiting.pop(0))
        row=[j.name for j in active];timeline.append(dict(tick=tick,active=row))
        for j in active:
            remaining[j.name]-=1;first.setdefault(j.name,tick+1)
            if remaining[j.name]==0:finish[j.name]=tick+1
        active=[j for j in active if remaining[j.name]>0];tick+=1
    records=[dict(name=j.name,arrival=j.arrival,first_token=first[j.name],finish=finish[j.name],
                  ttft=first[j.name]-j.arrival,latency=finish[j.name]-j.arrival,tokens=j.tokens) for j in jobs]
    return dict(makespan=tick,tokens=sum(j.tokens for j in jobs),capacity_slots=tick*slots,
                active_fraction=sum(j.tokens for j in jobs)/(tick*slots),records=records,timeline=timeline)


def demo():
    jobs=[Job('A',0,4),Job('B',0,1),Job('C',1,2),Job('D',2,1)]
    return dict(jobs=[vars(j) for j in jobs],static=simulate(jobs,continuous=False),continuous=simulate(jobs),
                scope='one output token per active request per unit tick; no prefill, variable step cost or hardware timing')
