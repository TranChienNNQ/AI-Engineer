"""Explicit hypothetical ledgers, not hardware counters."""
import math


def roofline(flops,bytes_moved,peak_flops,bandwidth):
    if not all(math.isfinite(v) and v>0 for v in [flops,bytes_moved,peak_flops,bandwidth]):raise ValueError('positive finite quantities required')
    intensity=flops/bytes_moved
    return dict(intensity=intensity,ridge=peak_flops/bandwidth,compute_s=flops/peak_flops,memory_s=bytes_moved/bandwidth,
        lower_bound_s=max(flops/peak_flops,bytes_moved/bandwidth),attainable_flops=min(peak_flops,bandwidth*intensity))


def amdahl(fraction,speedup):
    if not 0<=fraction<=1 or not math.isfinite(speedup) or speedup<=0:raise ValueError('fraction/speedup')
    return 1/(1-fraction+fraction/speedup)


def sectors(stride_words,offset_words=0,lanes=32):
    if type(stride_words) is not int or stride_words<1 or type(offset_words) is not int or offset_words<0 or type(lanes) is not int or lanes<1:raise ValueError('address model')
    # 4-byte words, aligned base, 32-byte sectors; no cache reuse/architecture simulation.
    return len({((offset_words+i*stride_words)*4)//32 for i in range(lanes)})


def hand():
    n=1_000_000
    # Count mul + add as 2 FLOP/element; ReLU comparison excluded explicitly.
    return dict(fusion=dict(elements=n,separate_bytes=24*n,fused_bytes=8*n,flops=2*n,ideal_traffic_ratio=3),
        roofline=roofline(2*n,8*n,100e12,1e12),
        tiled_matmul=dict(flops_convention=2*4**3,scalar_operand_bytes=2*4**3*4,tile2_operand_bytes=2*(4//2)**3*2**2*4,
            output_bytes=4**2*4,note='source-level operand load ledger; output write once; no hardware cache effects'),
        sectors=dict(contiguous=sectors(1),offset1=sectors(1,1),stride8=sectors(8)),
        online=dict(scores=[0.,math.log(2),math.log(3)],values=[10.,20.,40.],m=math.log(3),l=2.,z=170/3,output=85/3),
        amdahl=dict(fraction=.4,kernel_speedup=3.,end_to_end_speedup=amdahl(.4,3)),
        compile_break_even=dict(compile_seconds=2.,saved_seconds_per_call=.0001,strictly_faster_after_calls=20_001))
