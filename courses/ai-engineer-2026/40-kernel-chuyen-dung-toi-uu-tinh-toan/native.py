"""Build only bundled C source. POSIX/macOS + a Clang/GCC-compatible cc required."""
import ctypes as ct
import hashlib
import platform
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np


class Native:
    def __init__(self,build):
        build=Path(build);build.mkdir(parents=True,exist_ok=True)
        cc=shutil.which('cc')
        if cc is None:raise RuntimeError('cc not found; install a C toolchain to run native lab')
        src=Path(__file__).with_name('kernels.c');lib=build/('kernels.dylib' if platform.system()=='Darwin' else 'kernels.so')
        if platform.system() not in ['Darwin','Linux']:raise RuntimeError('tested path requires macOS or Linux')
        flags=['-O2','-std=c11','-ffp-contract=off','-fno-fast-math']
        command=[cc,*flags,*(['-dynamiclib'] if platform.system()=='Darwin' else ['-shared','-fPIC']),str(src),'-o',str(lib)]
        begin=time.perf_counter_ns();p=subprocess.run(command,capture_output=True,text=True,timeout=60)
        if p.returncode:raise RuntimeError('C compilation failed: '+p.stderr)
        self.compile_ns=time.perf_counter_ns()-begin
        self.identity=dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),flags=flags,
            compiler=subprocess.run([cc,'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0])
        self.lib=ct.CDLL(str(lib.resolve()));ptr=ct.POINTER(ct.c_float)
        self.lib.fused.argtypes=[ptr,ptr,ct.c_size_t,ct.c_float,ct.c_float]
        self.lib.separate.argtypes=[ptr,ptr,ptr,ptr,ct.c_size_t,ct.c_float,ct.c_float]
        self.lib.fused.restype=self.lib.separate.restype=None
    def bind(self,x,y,a,b,mode='fused',scratch=None):
        if mode not in ['fused','separate']:raise ValueError('mode')
        buffers=[x,y]
        if mode=='separate':
            if scratch is None or len(scratch)!=2:raise ValueError('two scratch buffers required')
            buffers+=list(scratch)
        for i,v in enumerate(buffers):
            if not isinstance(v,np.ndarray) or v.dtype!=np.float32 or v.ndim!=1 or not v.flags.c_contiguous or not v.flags.aligned:raise ValueError('aligned contiguous float32 vector required')
            if v.shape!=x.shape or len(v)<1:raise ValueError('nonempty matching shape required')
            if i and not v.flags.writeable:raise ValueError('writable output/scratch required')
        if any(np.shares_memory(v,w) for i,v in enumerate(buffers) for w in buffers[:i]):raise ValueError('buffers must be disjoint')
        if not np.isfinite(x).all() or not np.isfinite([a,b]).all():raise ValueError('finite inputs required')
        a,b=np.float32(a),np.float32(b)
        with np.errstate(over='ignore',invalid='ignore'):
            finite=np.isfinite(x*a+b).all()
        if not finite:raise ValueError('FP32 affine overflow')
        ptr=ct.POINTER(ct.c_float);args=[v.ctypes.data_as(ptr) for v in buffers]+[len(x),ct.c_float(a),ct.c_float(b)]
        fn=getattr(self.lib,mode)
        # Closure holds every buffer alive. Validation/allocation occurs outside timed loop.
        def call():
            fn(*args)
            return y
        return call
