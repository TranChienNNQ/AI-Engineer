import hashlib
import numpy as np

SPEC=dict(version='daily-forecast-v1',samples=420,lookback=28,horizon=7,season=7,initial_end=224,validation_end=280,calibration_end=364,alpha=.2,start_date='2025-01-06')


def dataset(shift=True):
    t=np.arange(420);rng=np.random.default_rng(43);innov=rng.normal(size=420)
    plan=(((t+3)%35)<3).astype(float)
    trend=40+.05*t;season=8*np.sin(2*np.pi*t/7)+2*np.cos(2*np.pi*t/7)
    shock=np.where(t>=364,15+7*np.sin(2*np.pi*t/7+np.pi/3),0.) if shift else np.zeros(420)
    noise=np.zeros(420)
    for i in range(420):noise[i]=(.55*noise[i-1] if i else 0)+(3. if shift and i>=364 else 1.)*innov[i]
    y=np.maximum(0,np.round(trend+season+14*plan+noise+shock))
    return dict(t=t,y=y,plan=plan,plan_available_at=t-14,components=dict(trend=trend,season=season,plan_effect=14*plan,noise=noise,shock=shock),shift=shift)


def fingerprint(data):
    h=hashlib.sha256()
    for key in ['t','y','plan','plan_available_at']:h.update(key.encode());h.update(np.asarray(data[key]).tobytes())
    return h.hexdigest()


def origins(start,end):
    if type(start) is not int or type(end) is not int or start<28 or end<=start or (end-start)%7:raise ValueError('whole seven-day evaluation blocks required')
    return list(range(start-1,end-7,7))


def validate_grid(data):
    t=np.asarray(data['t']);y=np.asarray(data['y'])
    if t.ndim!=1 or y.shape!=t.shape or len(t)<35 or not np.array_equal(t,np.arange(len(t))):raise ValueError('complete ordered daily grid required; missing/duplicate days must be resolved explicitly')
    if not np.isfinite(y).all():raise ValueError('finite observations required; no implicit missing-to-zero')
    if np.asarray(data['plan']).shape!=t.shape or np.asarray(data['plan_available_at']).shape!=t.shape:raise ValueError('covariate shape')
