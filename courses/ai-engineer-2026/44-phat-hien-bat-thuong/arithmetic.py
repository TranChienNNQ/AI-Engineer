import numpy as np
from detectors import robust,threshold,cusum
from metrics import confusion,average_precision
from alerts import apply


def hand():
    x=np.array([10.,11.,12.,13.,14.]);med,scale=robust(x)
    v=np.array([1.,1.])/np.sqrt(2);z=np.array([2.,0.]);recon=(z@v)*v
    values,state=cusum([1.,1.,1.,1.],k=.5)
    q,rank=threshold(np.arange(1,20),.1)
    y=[0,1,0,1,0];scores=[.1,.9,.8,.7,.2]
    return dict(robust=dict(median=float(med),mad=float(np.median(abs(x-med))),scale=float(scale),new_value=20.,modified_z=float((20-med)/scale)),
        pca=dict(point=z.tolist(),projection=recon.tolist(),squared_residual=float(np.sum((z-recon)**2))),
        mahalanobis=dict(covariance=[[1,.9],[.9,1]],point=[1.,-1.],squared_distance=float(np.array([1.,-1.])@np.linalg.solve([[1,.9],[.9,1]],[1.,-1.]))),
        knn=dict(reference=[0.,1.,2.,10.],query=3.,k=2,mean_nearest_distance=1.5),
        threshold=dict(calibration_scores=list(range(1,20)),alpha=.1,rank=rank,value=q,strict_comparison='score > threshold'),
        cusum=dict(residual=[1.,1.,1.,1.],k=.5,score=values.tolist(),final_state=state),
        metrics=confusion(y,np.array(scores)>.75),ap=average_precision(y,scores),
        base_rate=dict(prevalence=.001,tpr=.9,fpr=.01,precision=(.001*.9)/(.001*.9+.999*.01)),
        alerts=apply([0,1,1,1,0,0,0,1,1,1,1,1],persist=2,clear=3,cooldown=4))
