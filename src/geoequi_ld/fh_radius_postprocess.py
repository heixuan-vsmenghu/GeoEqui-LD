"""Fixed B symmetric FH-radius postprocess on four supplied coordinate arrays.

This is the exact array-only combination used by the reported B experiment,
not a trained native Geo model and not the complete inference pipeline.
GEO_A and GEO_B supply two Geo predictions; STLD1801 and STLD1901 supply
STLD predictions. Every input has shape [N, 3, 2] in the same original-pixel
frame, ordered PSR, PSL, FH. Endpoint inference, native preprocessing,
coordinate inversion, input validation and clinical evaluation are not
provided here. No weights, images, labels or execution records are included.

``combine`` returns the four inputs, both coordinate ensembles and CANDIDATE,
plus fallback metadata. The candidate retains the averaged Geo PS points and
FH ray; its radius is the mean of the two STLD-FH distances to that Geo PSR.
Degenerate rays/radii retain the Geo ensemble. This fallback does not repair
nonfinite inputs or establish per-case nonharm or clinical validity.

The function and required constants are copied without numerical edits from
the fixed experiment implementation; tests use constructed arrays only.
"""
# ruff: noqa: E501, E701, E702
# Long/compact statements below preserve the executed function body verbatim.
from __future__ import annotations

import numpy as np

EPS=1e-8

MODEL_IDS=('GEO_A','GEO_B','STLD1801','STLD1901')

def require(ok,message):
    if not ok:raise ValueError(message)

def combine(points):
    require(set(points)==set(MODEL_IDS),'Exactly four fixed native outputs required')
    arrays={k:np.asarray(v,np.float64) for k,v in points.items()};shape=arrays['GEO_A'].shape
    require(len(shape)==3 and shape[1:]==(3,2) and all(v.shape==shape for v in arrays.values()),'Four original-pixel coordinate shapes differ')
    geo=(arrays['GEO_A']+arrays['GEO_B'])/2;stld=(arrays['STLD1801']+arrays['STLD1901'])/2;candidate=geo.copy()
    ray=geo[:,2]-geo[:,0];ray_length=np.hypot(ray[:,0],ray[:,1]);r1=arrays['STLD1801'][:,2]-geo[:,0];r2=arrays['STLD1901'][:,2]-geo[:,0]
    radius=(np.hypot(r1[:,0],r1[:,1])+np.hypot(r2[:,0],r2[:,1]))/2
    valid=np.isfinite(geo).all(axis=(1,2))&np.isfinite(radius)&np.isfinite(ray_length)&(ray_length>EPS)&(radius>EPS)
    candidate[valid,2]=geo[valid,0]+ray[valid]*(radius[valid]/ray_length[valid])[:,None]
    require(np.array_equal(candidate[:,:2],geo[:,:2],equal_nan=True),'Candidate PS points must be untouched Geo ensemble values')
    return dict(arrays,**{'GEO_ENS':geo,'STLD_ENS':stld,'CANDIDATE':candidate}),dict(fallback_count=int((~valid).sum()),fallback_mask=~valid,epsilon=EPS,PSR_PSL_exact_GEO_ENS=True)
