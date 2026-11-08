"""Warm CPU native microbenchmark; raw repeated samples, no performance pass/fail."""
import time
import numpy as np


def run(native):
    rng=np.random.default_rng(400);rows=[]
    for n,inner in [(256,200),(65_536,30),(1_048_576,5)]:
        x=rng.normal(size=n).astype(np.float32);y=np.empty_like(x);t=np.empty_like(x);u=np.empty_like(x)
        y_fused=np.empty_like(x)
        calls=dict(separate=native.bind(x,y,1.25,-.5,'separate',(t,u)),fused=native.bind(x,y_fused,1.25,-.5))
        expected=np.maximum(x*np.float32(1.25)+np.float32(-.5),0)
        for call in calls.values():call()
        assert np.array_equal(y,expected) and np.array_equal(y_fused,expected)
        for call in calls.values():
            for _ in range(5):call()
        samples={mode:[] for mode in calls};orders=[]
        for sample in range(9):
            order=['separate','fused'] if sample%2==0 else ['fused','separate'];orders.append(order)
            for mode in order:
                begin=time.perf_counter_ns()
                for _ in range(inner):calls[mode]()
                samples[mode].append((time.perf_counter_ns()-begin)/inner)
        modes={}
        for mode,times in samples.items():
            p10,median,p90=np.quantile(times,[.1,.5,.9]);modes[mode]=dict(samples_ns=times,p10_ns=float(p10),median_ns=float(median),p90_ns=float(p90))
        rows.append(dict(elements=n,inner=inner,warmup_calls_per_mode=5,sample_orders=orders,modes=modes,
            median_ratio_separate_over_fused=modes['separate']['median_ns']/modes['fused']['median_ns'],max_error=0.))
    return dict(rows=rows,compile_ns=native.compile_ns,identity=native.identity,
        scope='warm repeated CPU buffers; Python loop + ctypes dispatch + C compute; excludes validation, allocation and compilation; one native thread, CPU affinity/frequency not controlled; no HBM traffic measurement')
