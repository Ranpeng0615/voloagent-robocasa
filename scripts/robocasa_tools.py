"""Perception queries and single-action robot tools, orchestrated by the VLM.

SAM3 and GraspGen predict geometry without moving the robot. Motion tools track
one requested pose or operate the gripper; no grasp/place sequence is inserted.
Only calibrated cameras and robot proprioception come from the simulator.
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
from vlm_orchestrator.grasp.tool import LIBERO_GRASPGEN_TO_EE_YAW_RAD
from duplex_contract import validate_tool_args
from whole_body import whole_body_action


@dataclass
class Waypoint:
    phase: str
    pos: np.ndarray
    rot: np.ndarray
    close: float
    dwell_s: float = 0.
    timeout_s: float | None = None


@dataclass
class ToolPlan:
    name: str
    waypoints: list[Waypoint]
    evidence: dict


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


def perceive(guide, snapshot, log, client=None):
    client = client or GraspClient('http://127.0.0.1:8003',timeout=40)
    name = guide['tool']
    if name not in {'sam3','graspgen'}:raise ValueError('Not a perception tool: '+name)
    validate_tool_args(name, guide['args'])
    target = guide['args']['target']
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
    center, _ = localize(snapshot,mask)
    log.emit('sam3_result',tool=name,target=target,score=score,mask_pixels=int(mask.sum()),box=box,
             duration_s=time.monotonic()-start,revision=guide['revision'])
    evidence = {'target':target,'camera':camera,'sam3_score':score,'mask_pixels':int(mask.sum()),'box':list(box),
                'observed_step':snapshot['step'],'perception_world_center':center.tolist(),
                'motion_solver':'whole_body_differential_ik' if 'kinematics' in snapshot else 'arm_osc'}
    artifacts=Path(os.environ['RUN_DIR'])/'tool_assets';artifacts.mkdir(exist_ok=True)
    prefix=artifacts/f'query_{guide["request_id"]}_{name}'
    np.savez_compressed(prefix.with_suffix('.npz'),rgb=snapshot['left'],mask=mask,
        depth_m=snapshot['depth'],K=snapshot['K'],T_world_camera=snapshot['T_camera'])
    evidence.update(tool=name, request_id=guide['request_id'],
        observed_wall_s=snapshot.get('wall_s'),
        captured_at_monotonic=snapshot.get('captured_at_monotonic'))
    if name == 'graspgen':
        k = snapshot['K']; cloud = scene_cloud(snapshot)
        gravity = snapshot['T_camera'][:3,:3].T@np.array([0.,0.,-1.])
        tick=time.monotonic()
        pose, confidence = client.compute_grasp(cloud,mask,mask.shape,k[0,0],
            fy_px=k[1,1],cx_px=k[0,2],cy_px=k[1,2],topdown_gravity=gravity,
            topdown_threshold=.65,enable_topdown_filter=guide['args'].get('topdown_only',False))
        world = snapshot['T_camera']@pose
        hand = world.copy()
        # Same GraspGen-to-Panda hand axis correction as the author's executor.
        hand[:3,:3] = world[:3,:3]@Rotation.from_euler('z',LIBERO_GRASPGEN_TO_EE_YAW_RAD).as_matrix()
        # RoboCasa controls the grip SITE, not the Panda hand flange. Measure
        # this rigid robot-only transform from its model, rather than guessing.
        ee = hand@snapshot['T_hand_to_site']
        log.emit('graspgen_result',confidence=confidence,duration_s=time.monotonic()-tick,
                 revision=guide['revision'],grasp_world=world.tolist(),ee_target=ee.tolist())
        evidence.update(graspgen_confidence=float(confidence), grasp_world=world.tolist(),
            pose_ref=f"grasp:{guide['request_id']}",
            position_world=ee[:3,3].tolist(),
            quaternion_xyzw=Rotation.from_matrix(ee[:3,:3]).as_quat().tolist(),
            approach_axis_world=world[:3,2].tolist())
    prefix.with_suffix('.json').write_text(json.dumps(evidence,indent=2))
    return evidence


def make_plan(guide, snapshot, log, grasp_pose=None):
    """Translate one VLM-selected robot action, without adding other actions."""
    name,args=guide['tool'],guide['args']
    validate_tool_args(name,args)
    p,r=snapshot['ee_pos'].copy(),snapshot['ee_rot'].copy()
    close=float(snapshot['gripper_command'])
    if name=='gripper':
        close=1. if args['command']=='close' else 0.
        return ToolPlan(name,[Waypoint('gripper',p,r,close,dwell_s=args.get('duration_s',.5))],
            {'command':args['command'],'duration_s':args.get('duration_s',.5)})
    if name!='move_ee':
        raise ValueError('Not a motion tool: '+name)
    ref=args.get('pose_ref')
    if ref and ref!='current':
        if grasp_pose is None or grasp_pose.get('pose_ref')!=ref:
            raise ValueError('Unknown grasp pose reference; use a returned GraspGen pose_ref')
        p=np.asarray(grasp_pose['position_world'],dtype=float)
        r=Rotation.from_quat(grasp_pose['quaternion_xyzw']).as_matrix()
    if 'position_world' in args:p=np.asarray(args['position_world'],dtype=float)
    if 'quaternion_xyzw' in args:r=Rotation.from_quat(args['quaternion_xyzw']).as_matrix()
    p=p+np.asarray(args.get('offset_world',[0.,0.,0.]))+r@np.asarray(args.get('offset_local',[0.,0.,0.]))
    if args.get('gripper') in {'open','close'}:close=1. if args['gripper']=='close' else 0.
    evidence={'args':args,'position_world':p.tolist(),
        'quaternion_xyzw':Rotation.from_matrix(r).as_quat().tolist(),
        'gripper_command':close,'timeout_s':args.get('timeout_s')}
    log.emit('motion_request',tool=name,revision=guide['revision'],**evidence)
    return ToolPlan(name,[Waypoint('move_ee',p,r,close,timeout_s=args.get('timeout_s'))],evidence)


class MotionExecutor:
    def __init__(self, plan):
        self.plan = plan
        self.index = 0
        self.started = time.monotonic()
        self.reached = None
        self.status = 'moving'
        self.tracking_error = {}
        self.ik_diagnostics={}
        self.base_anchor=None
        self.gripper_issued=False

    @property
    def phase(self):
        return self.plan.waypoints[self.index].phase if self.index < len(self.plan.waypoints) else self.status

    def tick(self, snapshot):
        if self.index >= len(self.plan.waypoints):
            return None
        goal = self.plan.waypoints[self.index]
        if goal.phase=='gripper':
            if self.gripper_issued and time.monotonic()-self.started>=goal.dwell_s:
                self.index=len(self.plan.waypoints);self.status='motion_done_unverified'
                return None
            self.gripper_issued=True
            action=np.zeros(12);action[6]=goal.close
            return action
        dp = goal.pos-snapshot['ee_pos']
        dr_world = Rotation.from_matrix(goal.rot@snapshot['ee_rot'].T).as_rotvec()
        self.tracking_error={**self.ik_diagnostics,'position_m':float(np.linalg.norm(dp)),
            'rotation_rad':float(np.linalg.norm(dr_world))}
        position_tolerance=.009
        rotation_tolerance=.12
        close_enough = np.linalg.norm(dp) < position_tolerance and np.linalg.norm(dr_world) < rotation_tolerance
        now = time.monotonic()
        if close_enough:
            self.reached = self.reached or now
            if now-self.reached >= goal.dwell_s:
                self.index += 1; self.started=now; self.reached=None
                if self.index == len(self.plan.waypoints):
                    self.status='motion_done_unverified'
                    if snapshot.get('gripper_command',goal.close)!=goal.close:
                        action=np.zeros(12);action[6]=goal.close
                        return action
                    return None
                return self.tick(snapshot)
        else:
            self.reached=None
        if goal.timeout_s is not None and now-self.started > goal.timeout_s:
            self.status='failed_'+goal.phase
            self.index=len(self.plan.waypoints)
            return None
        if 'kinematics' in snapshot:
            if self.base_anchor is None:self.base_anchor=snapshot['kinematics']['q'][:3].copy()
            action,diagnostics=whole_body_action(snapshot,dp,dr_world,goal.close,self.base_anchor,
                base_enabled=True)
            self.tracking_error.update(diagnostics)
            self.ik_diagnostics=diagnostics
            return action
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
