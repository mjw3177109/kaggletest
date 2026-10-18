"""V2.7 CPU-only paired OOF fusion mathematics (no target leakage).

Blends *independent out-of-fold predictions* from already-trained Kaggle
V2.5 and V2.6 pipelines; never trains on or uses the competition test labels.
Do not select among hundreds of weights on the same OOF labels.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def model_oof(payload, chosen: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert archived 3-fold OOF components into the model actually selected."""
    l = np.asarray(payload["oof_lgb"], dtype=np.float64).reshape(-1)
    x = np.asarray(payload["oof_xgb"], dtype=np.float64).reshape(-1)
    if l.shape != x.shape or not len(l):
        raise ValueError("OOF lengths do not match")
    if not np.isfinite(l).all() or not np.isfinite(x).all():
        raise ValueError("Invalid OOF probabilities")
    if (l.min() < 0 or l.max() > 1 or x.min() < 0 or x.max() > 1):
        raise ValueError("OOF probability outside [0,1]")
    if chosen == "lightgbm":
        return l, x
    if chosen == "xgboost":
        return x, l
    if chosen == "blend_mean":
        return (l + x) / 2, l
    if chosen == "blend_rank":
        return (rankdata(l, method="average") + rankdata(x, method="average")) / (2 * len(l)), l
    raise ValueError("Unsupported archived model choice: " + repr(chosen))


def paired_fusion(y25, y26, best25, best26, pred25, pred26, threshold=0.00010):
    y1=np.asarray(y25).reshape(-1)
    y2=np.asarray(y26).reshape(-1)
    if not np.array_equal(y1,y2) or len(y1)<10 or not np.isin(y1,[0,1]).all():
        raise ValueError("OOF targets differ or labels invalid; refuse to blend")
    a=np.asarray(best25,dtype=np.float64).reshape(-1)
    b=np.asarray(best26,dtype=np.float64).reshape(-1)
    if a.shape!=y1.shape or b.shape!=y1.shape:
        raise ValueError("OOF model predictions are not aligned")
    ta=np.asarray(pred25,dtype=np.float64).reshape(-1)
    tb=np.asarray(pred26,dtype=np.float64).reshape(-1)
    if ta.shape!=tb.shape or not len(ta):
        raise ValueError("Kaggle test files have different row counts")
    for row in (a,b,ta,tb):
        if not np.isfinite(row).all():
            raise ValueError("Nonfinite model prediction")
    # Fixed blend -- chosen independently of OOF scores. Rank each model on
    # the OOF train set and on Kaggle test separately (no y anywhere).
    fused_oof=(rankdata(a)+rankdata(b))/(2*len(a))
    fused_test=(rankdata(ta)+rankdata(tb))/(2*len(ta))
    score25=float(roc_auc_score(y1,a))
    score26=float(roc_auc_score(y1,b))
    score=float(roc_auc_score(y1,fused_oof))
    # The benchmark must be supplied separately using the actual declared
    # complete 3-fold CV, not just compared to these two selected components.
    if not (0 <= score <= 1 and 0 <= score25 <= 1 and 0 <= score26 <= 1):
        raise ValueError("Unexpected AUC")
    return {
        "score25":score25,
        "score26":score26,
        "fused_cv_auc":score,
        "gain_over_best_component":score-max(score25,score26),
        "threshold":threshold,
        "fused_test":fused_test,
    }
