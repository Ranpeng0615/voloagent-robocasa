"""Author's SAM3/GraspGen perception with a native RoboCasa motion adapter.

Only calibrated cameras and robot proprioception come from the simulator.
No simulator object identities, poses, contact flags or segmentation are read.
The Cartesian path is closed-loop and cancellable; it is not cuRobo planning.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from vlm_orchestrator.grasp.client import GraspClient
from vlm_orchestrator.grasp.camera import CameraIntrinsics, depth_to_pointcloud
from vlm_orchestrator.grasp.tool import (
    LIBERO_GRASPGEN_TO_EE_YAW_RAD, PRE_GRASP_OFFSET, RETREAT_HEIGHT,
)


@dataclass
class Waypoint:
    phase: str
    pos: np.ndarray
    rot: np.ndarray
    close: float
    dwell_s: float = 0.
    timeout_s: float = 4.


@dataclass
class ToolPlan:
    name: str
    waypoints: list[Waypoint]
    evidence: dict
    held_offset: np.ndarray | None = None


def scene_cloud(snapshot):
    k = snapshot['K']
    h,w = snapshot['depth'].shape
    intr = CameraIntrinsics(width=w,height=h,fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2])
    return depth_to_pointcloud(snapshot['depth'], intr, max_depth=3.)


def localize(snapshot, mask):
    h,w = mask.shape
    v,u = np.nonzero(mask & np.isfinite(snapshot['depth']) & (snapshot['depth'] > 0) & (snapshot['depth'] < 3))
    if len(u) < 20:
        raise ValueError('Too few valid masked depth pixels')
    z = snapshot['depth'][v,u]
    k = snapshot['K']
    points = np.stack([(u-k[0,2])/k[0,0]*z,(v-k[1,2])/k[1,1]*z,z],axis=1)
    world = points@snapshot['T_camera'][:3,:3].T+snapshot['T_camera'][:3,3]
    return np.median(world,axis=0), world


def make_plan(guide, snapshot, log, held_offset=None, client=None):
    client = client or GraspClient('http://127.0.0.1:8003',timeout=40)
    name = guide['tool']
    target = guide['args']['target' if name == 'grasp' else 'destination']
    start = time.monotonic()
    # The arm can occlude an external camera while the wrist sees the object.
    # Try real calibrated views, keeping each RGB/mask/depth/pose pair aligned.
    failures=[]
    views=snapshot.get('views',{'left':{'rgb':snapshot['left'],'depth':snapshot['depth'],
        'K':snapshot['K'],'T_camera':snapshot['T_camera']}})
    for camera,view in views.items():
        try:
            mask,score,box=client.detect_and_segment(view['rgb'],target)
            snapshot={**snapshot,**view,'left':view['rgb']}
            break
        except Exception as exc:
            failures.append(camera+': '+str(exc))
            log.emit('sam3_view_failed',camera=camera,revision=guide['revision'],error=str(exc))
    else:
        raise ValueError('No usable SAM3 target in available cameras: '+'; '.join(failures))
    center, obj_world = localize(snapshot,mask)
    log.emit('sam3_result',tool=name,target=target,score=score,mask_pixels=int(mask.sum()),box=box,
             duration_s=time.monotonic()-start,revision=guide['revision'])
    evidence = {'target':target,'camera':camera,'sam3_score':score,'mask_pixels':int(mask.sum()),'box':list(box),
                'observed_step':snapshot['step'],'perception_world_center':center.tolist()}
    artifacts=Path(os.environ['RUN_DIR'])/'tool_assets';artifacts.mkdir(exist_ok=True)
    prefix=artifacts/f'revision_{guide["revision"]}_{name}'
    np.savez_compressed(prefix.with_suffix('.npz'),rgb=snapshot['left'],mask=mask,
        depth_m=snapshot['depth'],K=snapshot['K'],T_world_camera=snapshot['T_camera'])
    p,r = snapshot['ee_pos'],snapshot['ee_rot']
    if name == 'grasp':
        k = snapshot['K']; cloud = scene_cloud(snapshot)
        gravity = snapshot['T_camera'][:3,:3].T@np.array([0.,0.,-1.])
        tick=time.monotonic()
        pose, confidence = client.compute_grasp(cloud,mask,mask.shape,k[0,0],
            fy_px=k[1,1],cx_px=k[0,2],cy_px=k[1,2],topdown_gravity=gravity,
            topdown_threshold=.65,enable_topdown_filter=True)
        world = snapshot['T_camera']@pose
        hand = world.copy()
        # Same GraspGen-to-Panda hand axis correction as the author's executor.
        hand[:3,:3] = world[:3,:3]@Rotation.from_euler('z',LIBERO_GRASPGEN_TO_EE_YAW_RAD).as_matrix()
        # RoboCasa controls the grip SITE, not the Panda hand flange. Measure
        # this rigid robot-only transform from its model, rather than guessing.
        ee = hand@snapshot['T_hand_to_site']
        # Parallel-jaw grasps are equivalent under a 180-degree roll about the
        # approach axis. Choose the equivalent hand pose with less wrist twist.
        alternate=hand.copy();alternate[:3,:3]=hand[:3,:3]@Rotation.from_euler('z',np.pi).as_matrix()
        alternate_ee=alternate@snapshot['T_hand_to_site']
        if np.linalg.norm(Rotation.from_matrix(alternate_ee[:3,:3]@r.T).as_rotvec()) < np.linalg.norm(Rotation.from_matrix(ee[:3,:3]@r.T).as_rotvec()):
            ee=alternate_ee
        pos,rot = ee[:3,3],ee[:3,:3]
        approach = world[:3,2]
        pre = pos-approach*PRE_GRASP_OFFSET
        clear = p.copy();clear[2] = max(p[2],pre[2]+.06)
        above = pre.copy();above[2] = clear[2]
        log.emit('graspgen_result',confidence=confidence,duration_s=time.monotonic()-tick,
                 revision=guide['revision'],grasp_world=world.tolist(),ee_target=ee.tolist())
        evidence.update(graspgen_confidence=confidence,grasp_world=world.tolist())
        prefix.with_suffix('.json').write_text(json.dumps(evidence,indent=2))
        offset = rot.T@(center-pos)
        return ToolPlan(name,[Waypoint('clear',clear,r,0),Waypoint('above_target',above,rot,0),
            Waypoint('pregrasp',pre,rot,0),Waypoint('approach',pos,rot,0),
            Waypoint('close',pos,rot,1,dwell_s=.65),
            Waypoint('lift',pos+np.array([0,0,RETREAT_HEIGHT]),rot,1)],evidence,offset)
    if held_offset is None:
        raise ValueError('No held-object offset available; use VLA placement or grasp first')
    # Destination comes from visible support depth, not simulator object poses.
    # Place above the observed support and release; visual supervision verifies.
    object_offset = r@held_offset
    release = center-object_offset
    release[2] = np.quantile(obj_world[:,2],.8)-object_offset[2]+.055
    clear = p.copy();clear[2] = max(p[2],release[2]+.12)
    above = release.copy();above[2] = clear[2]
    return ToolPlan(name,[Waypoint('lift_for_transport',clear,r,1),
        Waypoint('transport',above,r,1),Waypoint('lower',release,r,1),
        Waypoint('release',release,r,0,dwell_s=.5),
        Waypoint('retreat',release+np.array([0,0,.12]),r,0)],evidence)


class MotionExecutor:
    def __init__(self, plan):
        self.plan = plan
        self.index = 0
        self.started = time.monotonic()
        self.reached = None
        self.status = 'moving'
        self.tracking_error = {}

    @property
    def phase(self):
        return self.plan.waypoints[self.index].phase if self.index < len(self.plan.waypoints) else self.status

    def tick(self, snapshot):
        if self.index >= len(self.plan.waypoints):
            return None
        goal = self.plan.waypoints[self.index]
        dp = goal.pos-snapshot['ee_pos']
        dr_world = Rotation.from_matrix(goal.rot@snapshot['ee_rot'].T).as_rotvec()
        self.tracking_error={'position_m':float(np.linalg.norm(dp)),
            'rotation_rad':float(np.linalg.norm(dr_world))}
        position_tolerance=.009 if goal.phase in {'approach','close','lower','release'} else .020
        rotation_tolerance=.12 if goal.phase in {'approach','close','lower','release'} else .20
        close_enough = np.linalg.norm(dp) < position_tolerance and np.linalg.norm(dr_world) < rotation_tolerance
        now = time.monotonic()
        if close_enough:
            self.reached = self.reached or now
            if now-self.reached >= goal.dwell_s:
                self.index += 1; self.started=now; self.reached=None
                if self.index == len(self.plan.waypoints):
                    self.status='motion_done_unverified'
                    return None
                return self.tick(snapshot)
        else:
            self.reached=None
        if now-self.started > goal.timeout_s:
            self.status='failed_'+goal.phase
            self.index=len(self.plan.waypoints)
            return None
        # Native OSC takes deltas in the controller's BASE reference frame.
        origin=snapshot['controller_origin_rot']
        delta_pos=origin.T@dp
        delta_rot=origin.T@dr_world
        delta=np.r_[np.clip(delta_pos,-.012,.012),np.clip(delta_rot,-.10,.10)]
        scale=snapshot['controller_scale']
        action=np.zeros(12)
        action[:6]=np.clip(delta/scale,-1,1)
        action[6]=goal.close
        return action
