import numpy as np


def top1(scores,labels,candidate_labels):
    s=np.asarray(scores);y=np.asarray(labels);cy=np.asarray(candidate_labels)
    if s.ndim!=2 or min(s.shape)<1 or s.shape!=(len(y),len(cy)) or not np.isfinite(s).all():raise ValueError('score/label shape')
    prediction=cy[s.argmax(axis=1)]
    return dict(correct=int(np.sum(prediction==y)),total=len(y),accuracy=float(np.mean(prediction==y)))


def late_fuse(image_scores=None,audio_scores=None,image_weight=.5):
    if not np.isfinite(image_weight) or not 0<=image_weight<=1:raise ValueError('fusion weight')
    if image_scores is None and audio_scores is None:raise ValueError('at least one modality required')
    for value in [image_scores,audio_scores]:
        if value is not None and (np.asarray(value).ndim!=2 or min(np.asarray(value).shape)<1 or not np.isfinite(value).all()):raise ValueError('finite nonempty score matrix required')
    if image_scores is None:return np.array(audio_scores,copy=True)
    if audio_scores is None:return np.array(image_scores,copy=True)
    a,b=np.asarray(image_scores),np.asarray(audio_scores)
    if a.shape!=b.shape or a.ndim!=2 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('fusion score shape')
    return image_weight*a+(1-image_weight)*b
