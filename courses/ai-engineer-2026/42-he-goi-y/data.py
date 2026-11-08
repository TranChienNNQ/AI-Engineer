"""Synthetic implicit interactions. No impression/exposure or real users."""
import hashlib
import json
import numpy as np

SPEC=dict(version='synthetic-recs-v1',warm_users=48,warm_items=72,topics=6,dim=12,train_before=10,validation_time=10,test_time=11,exclude_seen=True)


def dataset():
    catalog=[dict(id=i,topic=i//12 if i<72 else (i-72)%6,released=0 if i<72 else 11 if i<78 else 12,
                  retired=11 if i==71 else None) for i in range(79)]
    events=[];validation=[];test=[]
    cold_targets=set(np.random.default_rng(4200).choice(48,12,replace=False).tolist())
    for u in range(48):
        topic=u%6;rng=np.random.default_rng(420+topic)
        primary=np.roll(rng.permutation(12),2*(u//6))+12*topic
        secondary=np.random.default_rng(42000+u).choice(12,2,replace=False)+12*((topic+1)%6)
        history=[primary[0],primary[1],secondary[0],primary[2],primary[3],secondary[1],primary[4],primary[5]]
        events += [dict(user=u,item=int(item),time=t,event='click') for t,item in enumerate(history)]
        val=int(primary[6]);events.append(dict(user=u,item=val,time=10,event='click'))
        validation.append(dict(user=u,time=10,target=val,cohort='warm'))
        target=72+topic if u in cold_targets else next(int(i) for i in primary[7:] if i!=71)
        test.append(dict(user=u,time=11,target=target,cohort='new_item' if u in cold_targets else 'warm'))
        events.append(dict(user=u,item=target,time=11,event='click'))
    for topic in range(6):
        u=48+topic;target=topic*12
        test.append(dict(user=u,time=11,target=target,cohort='new_user'));events.append(dict(user=u,item=target,time=11,event='click'))
    events.sort(key=lambda r:(r['time'],r['user'],r['item']))
    return dict(spec=dict(SPEC),catalog=catalog,events=events,validation=validation,test=test)


def fingerprint(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def train_events(data):return [e for e in data['events'] if e['time']<SPEC['train_before']]


def seen_at(data,user,time):return {e['item'] for e in data['events'] if e['user']==user and e['time']<time}


def available(item,time):return item['released']<=time and (item['retired'] is None or time<item['retired'])


def eligible(data,request,blocked=()):
    seen=seen_at(data,request['user'],request['time']);blocked=set(blocked)
    return [i['id'] for i in data['catalog'] if available(i,request['time']) and i['id'] not in seen and i['id'] not in blocked]


def negative_pools(data):
    train=train_events(data);positives={u:{e['item'] for e in train if e['user']==u} for u in range(48)}
    # Snapshot at train boundary; never inspect validation/test targets to remove future positives.
    catalog=[i['id'] for i in data['catalog'] if available(i,9)]
    return {u:np.array([i for i in catalog if i not in positives[u]],dtype=np.int64) for u in positives}
