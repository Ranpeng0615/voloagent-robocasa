"""Warm model kernels with generic input; results are never used by an episode."""
import io,json,time
from pathlib import Path
import os
import numpy as np
import httpx
import imageio.v2 as imageio
import requests
from vlm_orchestrator.grasp.client import GraspClient
client=GraspClient('http://127.0.0.1:8003',timeout=240)
root=Path(os.environ['ROBOCASA_ROOT'])
report={'health':client.health()}
tick=time.monotonic()
preview=root/'cache/bread-and-cheese-initial.jpg'
image=imageio.imread(preview) if preview.exists() else np.full((256,256,3),127,np.uint8)
try:
    mask,score,box=client.detect_and_segment(image,'cutting board')
    report['sam3_pixels']=int(mask.sum())
except requests.HTTPError as exc:
    # A neutral synthetic image may contain no target; its model kernels still
    # execute. Other server failures are not accepted as successful warmup.
    if exc.response is None or 'SAM3 found no detections' not in exc.response.text:
        raise
    report['sam3_no_detection_on_warmup']=True
report['sam3_s']=time.monotonic()-tick
rng=np.random.default_rng(53)
points=rng.normal(size=(2048,3));points/=np.linalg.norm(points,axis=1,keepdims=True)
points=points*.035+np.array([0,0,1.])
tick=time.monotonic()
pose,confidence=client.compute_grasp(points,np.ones((64,64),bool),(64,64),60.,enable_topdown_filter=False)
report['graspgen_s']=time.monotonic()-tick
report['grasp_finite']=bool(np.isfinite(pose).all())
tick=time.monotonic()
sparse_pose,_=client.compute_grasp(points[:201],np.ones((64,64),bool),(64,64),60.,enable_topdown_filter=False)
report['sparse_grasp_finite']=bool(np.isfinite(sparse_pose).all())
report['sparse_grasp_s']=time.monotonic()-tick
print(json.dumps(report),flush=True)
