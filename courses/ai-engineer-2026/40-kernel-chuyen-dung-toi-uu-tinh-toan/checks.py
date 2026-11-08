import math
import numpy as np
from algorithms import tiled_matmul,dense_attention,online_attention,demo
from arithmetic import hand,roofline,amdahl,sectors


def run(native):
    names=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        names.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,TypeError):check(name,True)
        else:raise AssertionError(name+' accepted')
    rng=np.random.default_rng(401)
    for n in [1,3,7,31,257,1003,65536]:
        x=rng.uniform(-10,10,n).astype(np.float32);guard=np.full(n+2,-123.5,dtype=np.float32);y=guard[1:-1]
        t=np.empty_like(x);u=np.empty_like(x);reference=np.maximum(x*np.float32(1.25)+np.float32(-.5),0)
        # Independent scalar two-rounding oracle, different loop implementation.
        scalar=np.array([max(np.float32(np.float32(1.25)*a)+np.float32(-.5),np.float32(0)) for a in x],dtype=np.float32)
        native.bind(x,y,1.25,-.5)();check(f'native fused n={n}',np.array_equal(y,scalar) and np.array_equal(y,reference))
        check(f'tail canary n={n}',guard[0]==guard[-1]==np.float32(-123.5))
        native.bind(x,y,1.25,-.5,'separate',(t,u))();check(f'native separate n={n}',np.array_equal(y,scalar))
    x=np.array([-1.,0.,1.,2.],dtype=np.float32);y=np.empty_like(x)
    native.bind(x,y,2.,1.)();check('hand affine',y.tolist()==[0.,1.,3.,5.])
    for label,fn in [
        ('dtype',lambda:native.bind(x.astype(np.float64),y,1,0)),
        ('strided',lambda:native.bind(np.arange(8,dtype=np.float32)[::2],y,1,0)),
        ('overlap',lambda:native.bind(x,x,1,0)),
        ('shape',lambda:native.bind(x,y[:2],1,0)),
        ('nan',lambda:native.bind(np.full(4,np.nan,dtype=np.float32),y,1,0)),
        ('overflow',lambda:native.bind(np.full(4,3e38,dtype=np.float32),y,2,0)),
        ('empty',lambda:native.bind(x[:0],y[:0],1,0)),
        ('mode',lambda:native.bind(x,y,1,0,'unknown')),
        ('scratch',lambda:native.bind(x,y,1,0,'separate'))]:reject('native rejects '+label,fn)
    read_only=y.copy();read_only.setflags(write=False)
    reject('native rejects readonly',lambda:native.bind(x,read_only,1,0))
    for m,k,n,tile in [(1,1,1,4),(2,3,2,2),(5,7,3,2),(9,3,7,4)]:
        a=rng.normal(size=(m,k));b=rng.normal(size=(k,n));got,stats=tiled_matmul(a,b,tile)
        scalar=np.array([[sum(float(a[i,h])*float(b[h,j]) for h in range(k)) for j in range(n)] for i in range(m)])
        check(f'tiled matmul {m,k,n,tile}',np.allclose(got,scalar,rtol=1e-12,atol=1e-12))
        check(f'tile count {m,k,n,tile}',stats['tile_products']==math.ceil(m/tile)*math.ceil(n/tile)*math.ceil(k/tile))
    for m,n,d,dv,qt,kt in [(1,1,1,1,2,2),(3,7,4,2,2,3),(7,11,4,3,3,4),(9,5,3,4,4,2)]:
        q=rng.normal(size=(m,d));k=rng.normal(size=(n,d));v=rng.normal(size=(n,dv))
        mask=rng.random((m,n))>.4;mask[0]=False
        for allowed in [None,mask]:
            actual,stats=online_attention(q,k,v,allowed,qt,kt);expected=dense_attention(q,k,v,allowed)
            check(f'attention parity {m,n,d,allowed is None}',np.allclose(actual,expected,rtol=1e-12,atol=1e-12))
            check(f'attention tile bound {m,n,d,allowed is None}',stats['max_score_elements']<=min(qt,m)*min(kt,n))
        check(f'all masked zero {m,n,d}',np.array_equal(actual[0],np.zeros(dv)))
    q=np.array([[1.]]);k=np.array([[1000.],[1001.],[999.]]);v=np.array([[1.],[2.],[3.]])
    got,_=online_attention(q,k,v,query_tile=1,key_tile=1)
    check('stable large scores',np.isfinite(got).all() and np.allclose(got,dense_attention(q,k,v),rtol=1e-12,atol=1e-12))
    q=np.array([[1.]]);k=np.array([[0.],[math.log(2)],[math.log(3)]]);v=np.array([[10.],[20.],[40.]])
    got,_=online_attention(q,k,v,query_tile=1,key_tile=1);check('online hand',abs(got.item()-85/3)<1e-12)
    for label,fn in [
        ('matmul shape',lambda:tiled_matmul([[1,2]],[[1]])),
        ('tile zero',lambda:tiled_matmul([[1]],[[1]],0)),
        ('QK shape',lambda:online_attention([[1,2]],[[1]],[[1]])),
        ('mask shape',lambda:online_attention([[1]],[[1]],[[1]],np.ones((1,2),dtype=bool))),
        ('mask dtype',lambda:online_attention([[1]],[[1]],[[1]],np.ones((1,1)))),
        ('nonfinite',lambda:dense_attention([[np.nan]],[[1]],[[1]])),
        ('key tile',lambda:online_attention([[1]],[[1]],[[1]],key_tile=0)),
        ('roofline zero',lambda:roofline(1,0,1,1)),
        ('amdahl fraction',lambda:amdahl(1.1,2)),
        ('sector stride',lambda:sectors(0))]:reject(label,fn)
    h=hand();check('traffic ledger',h['fusion']['separate_bytes']==3*h['fusion']['fused_bytes'])
    check('roofline 8us',math.isclose(h['roofline']['lower_bound_s'],8e-6))
    check('sector counts',h['sectors']==dict(contiguous=4,offset1=5,stride8=32))
    check('Amdahl',math.isclose(amdahl(.4,3),15/11))
    check('Amdahl unaffected',amdahl(0,100)==1 and amdahl(1,3)==3)
    check('tile ledger',h['tiled_matmul']['scalar_operand_bytes']==512 and h['tiled_matmul']['tile2_operand_bytes']==256)
    r=demo();check('demo matmul',r['matmul']['output']==[[4.,5.],[10.,11.]])
    check('demo zero row',r['attention']['all_masked_row']==[0.,0.,0.])
    return dict(passed=len(names),checks=len(names),names=names)
