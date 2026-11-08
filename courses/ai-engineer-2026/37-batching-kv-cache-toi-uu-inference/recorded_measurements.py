"""Recorded local CPU samples from the run quoted in lesson 37, 06/11/2026.
Python3.14.6, torch2.15.0, macOS arm64, CPU1thread; untrained20992param model.
Static publication data, not a performance target. Re-run benchmark.py for your machine.
"""
import statistics
from scheduling import demo

SAMPLES_MS = {'single_full': [27.109834, 27.162417, 27.111917, 27.247958, 26.897166, 27.518583, 27.058709],
 'single_cache': [21.038959, 21.548125, 21.850167, 21.217375, 21.368375, 20.936333, 20.985167],
 'batch_cache': [6.269625, 6.8285, 6.627667, 6.525292, 6.706584, 6.73725, 6.904],
 'batch_cache_sdpa': [6.33175, 5.9075, 6.139583, 6.742791, 6.6845, 5.881916, 6.230292]}

TOKEN_PROBE = {'first_token_ms': 0.463375,
 'itl_ms': [0.305792,
            0.331792,
            0.312166,
            0.296292,
            0.29925,
            0.346292,
            0.340666,
            0.312417,
            0.328083,
            0.32475,
            0.316959,
            0.297541,
            0.299917,
            0.299375,
            0.2925],
 'scope': 'one local cached request; initial pack/cache allocation excluded; decode packing and '
          'selection included; no queue/network/tokenizer'}

def figure_data():
    rows=[]
    for name,samples in SAMPLES_MS.items():
        median=statistics.median(samples)
        rows.append(dict(name=name,samples_ms=samples,median_ms=median,min_ms=min(samples),max_ms=max(samples),output_tokens=64,output_tokens_per_second=64/(median/1000)))
    return dict(benchmark=dict(rows=rows),schedule=demo())
