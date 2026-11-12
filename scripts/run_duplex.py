"""Run a native RoboCasa episode with independent VLM/VLA/tool workers.

MuJoCo is owned exclusively by the main thread. Workers receive immutable
snapshots and return revision-tagged results; they cannot move the robot.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
from pathlib import Path
import queue
import random
import signal
import threading
import time

import imageio.v2 as imageio
import numpy as np
import gymnasium as gym
import robocasa
from openpi_client import image_tools, websocket_client_policy
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
from duplex_core import EventLog, Exchange, TimedVideo
from duplex_contract import executor_feedback, resolve_decision, MOTION_TOOLS
from duplex_vlm import supervise
from robocasa_tools import make_plan, perceive, MotionExecutor
from whole_body import RobotKinematics


def policy_wire(snapshot, instruction):
    return {'observation/image':image_tools.resize_with_pad(snapshot['left'],224,224),
        'observation/right_image':image_tools.resize_with_pad(snapshot['right'],224,224),
        'observation/wrist_image':image_tools.resize_with_pad(snapshot['wrist'],224,224),
        'observation/state':snapshot['state'], 'prompt':instruction}


def policy_worker(exchange, jobs, results, log):
    client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',8000)
    while not exchange.stop.is_set():
        try: guide,snapshot=jobs.get(timeout=.1)
        except queue.Empty: continue
        if exchange.stop.is_set(): break
        tick=time.monotonic()
        log.emit('vla_begin',revision=guide['revision'],observed_step=snapshot['step'])
        try:
            response=client.infer(policy_wire(snapshot,guide['instruction']))
            actions=np.asarray(response['actions'])
            if actions.ndim!=2 or actions.shape[1]!=12 or not np.isfinite(actions).all():
                raise ValueError('Invalid VLA action chunk')
            current,_=exchange.read()
            log.emit('vla_end',revision=guide['revision'],observed_step=snapshot['step'],
                current_step=current['step'],duration_s=time.monotonic()-tick)
            results.put(('vla',guide,snapshot,actions,None))
        except Exception as exc:
            log.emit('vla_error',error=f'{type(exc).__name__}: {exc}')
            results.put(('vla',guide,snapshot,None,str(exc)))


def tool_worker(exchange,jobs,results,log):
    while not exchange.stop.is_set():
        try: guide,snapshot,grasp_pose=jobs.get(timeout=.1)
        except queue.Empty: continue
        if exchange.stop.is_set(): break
        _,current=exchange.read()
        query=guide.get('query',False)
        if not query and current['revision']!=guide['revision']:
            results.put(('tool',guide,snapshot,None,'cancelled_before_start'))
            continue
        tick=time.monotonic()
        log.emit('tool_prepare_begin',revision=guide['revision'],tool=guide['tool'],observed_step=snapshot['step'])
        try:
            value=perceive(guide,snapshot,log) if query else make_plan(guide,snapshot,log,grasp_pose)
            results.put(('query' if query else 'tool',guide,snapshot,value,None))
            log.emit('tool_prepare_end',revision=guide['revision'],tool=guide['tool'],duration_s=time.monotonic()-tick)
        except Exception as exc:
            results.put(('query' if query else 'tool',guide,snapshot,None,f'{type(exc).__name__}: {exc}'))
            log.emit('tool_error',revision=guide['revision'],error=f'{type(exc).__name__}: {exc}')


class ExecutionState:
    """Control-thread-owned action state; no worker can bypass this gate."""
    def __init__(self, exchange, log):
        self.exchange,self.log=exchange,log
        self.actions=collections.deque()
        self.motion=None
        self.phase='awaiting_policy'
        self.last_close=0.
        self.grasp_pose=None
        self.latest_tool_result=None
        self.query_count=0
        self.vla_pending=False
        self.tool_pending=False
        self.stale_results=0
        self.applied_commands=0
        self.recent_events=collections.deque(maxlen=8)
        self.last_guidance_result=None
        self.last_visual_assessment=None

    def apply(self, command, step):
        _,guide=self.exchange.read()
        if command['based_revision']!=guide['revision']:
            self.last_guidance_result={'status':'rejected','reason':'obsolete_revision'}
            self.log.emit('command_discarded',reason='obsolete_revision',step=step,command=command)
            return None
        try:
            resolved=resolve_decision(command,guide,self.phase)
            if resolved['command']=='call_tool' and self.tool_pending:
                raise ValueError('Perception/motion preparation is still pending; see tool_pending')
        except ValueError as exc:
            self.last_guidance_result={'status':'rejected','reason':str(exc),'requested_command':command['command']}
            self.log.emit('command_rejected',step=step,command=command,reason=str(exc))
            return None
        kind=resolved['command']
        self.last_visual_assessment={k:resolved[k] for k in ['progress','grasp_status','stage','target','reason'] if k in resolved}
        self.last_visual_assessment['observed_step']=command.get('observed_step')
        self.last_guidance_result={'status':'applied','requested_command':command['command'],
            'effective_command':kind,'observed_step':command.get('observed_step'),
            'applied_step':step}
        if kind=='call_tool':
            self.query_count+=1
            self.tool_pending=True
            self.applied_commands+=1
            request={**resolved,'revision':guide['revision'],'request_id':self.query_count,'query':True}
            self.last_guidance_result.update(status='submitted',request_id=self.query_count)
            self.log.emit('tool_query_submitted',step=step,request=request)
            return request
        if kind=='continue':
            self.exchange.describe(resolved['instruction'])
            self.log.emit('continue',step=step,revision=guide['revision'],reason=command.get('reason'),
                instruction=resolved['instruction'])
            return None
        if kind=='stop':
            self.log.emit('stop_command',step=step,command=command)
            return 'stop'
        tool=resolved['tool']
        previous=guide['revision']
        guide=self.exchange.revise(tool,resolved['instruction'],resolved['args'])
        count=len(self.actions)
        self.actions.clear(); self.motion=None
        self.phase='holding' if tool=='hold' else 'preparing' if tool in MOTION_TOOLS else 'awaiting_policy'
        self.applied_commands+=1
        self.log.emit('command_applied',step=step,previous_revision=previous,revision=guide['revision'],
            cleared_actions=count,delivery_latency_s=time.monotonic()-command['received_at'],
            effective_command=kind,command=command)
        return guide

    def query_result(self, request, source, value, error, step):
        """Read-only results retain observation age; they never change motion."""
        self.tool_pending=False
        self.latest_tool_result={'request_id':request['request_id'],'tool':request['tool'],
            'args':request['args'],'observed_step':source['step'],
            'observed_wall_s':source.get('wall_s'),
            'status':'failed' if error else 'completed','error':error,'result':value}
        if not error and request['tool']=='graspgen':self.grasp_pose=value
        self.log.emit('tool_query_result',step=step,**self.latest_tool_result)

    def hold(self):
        action=np.zeros(12);action[6]=self.last_close
        return action

    def accepts(self, revision, kind, step):
        _,guide=self.exchange.read()
        if revision==guide['revision']:
            return True
        self.stale_results+=1
        self.log.emit('result_discarded',kind=kind,reason='obsolete_revision',revision=revision,step=step)
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',default='BreadAndCheese')
    parser.add_argument('--split',default='pretrain',choices=['pretrain','target'])
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--action-window',type=int,default=5)
    parser.add_argument('--video',action='store_true')
    args=parser.parse_args()
    random.seed(args.seed);np.random.seed(args.seed)
    run=Path(os.environ['RUN_DIR']);run.mkdir(parents=True,exist_ok=True)
    env=gym.make(f'robocasa/{args.task}',split=args.split,seed=args.seed,camera_depths=True)
    obs,info=env.reset()
    observed_at=time.monotonic()
    raw=env.unwrapped.env
    robot=raw.robots[0]
    c=robot.composite_controller.part_controllers['right']
    if c.input_type!='delta' or c.input_ref_frame!='base' or c._goal_update_mode!='achieved':
        raise RuntimeError('Unexpected native controller contract')
    hz=float(raw.control_freq);period=1/hz
    kinematics=RobotKinematics(robot,hz)
    horizon=int(get_task_horizon(args.task));limit=horizon/hz+20
    instruction=obs['annotation.human.task_description']
    log=EventLog(run/'duplex_events.jsonl')
    exchange=Exchange(instruction);execution=ExecutionState(exchange,log)
    # Start VLA immediately on the native task instruction. VLM supervises in
    # parallel and chooses individual perception or motion tools itself.
    jobs=queue.Queue(maxsize=1);tools=queue.Queue(maxsize=1);results=queue.Queue()
    report={'task':args.task,'split':args.split,'seed':args.seed,'instruction':instruction,
        'horizon':horizon,'control_hz':hz,'wall_limit_s':limit,'steps':0,'success':False,
        'wall_timeout':False,'checkpoint':'pi05_pretrain_human300','online_gt_state':False,
        'mode':'duplex','initial_executor':'vla','tool_selection':'vlm_orchestrated_primitives',
        'env_step_s':0.,'hold_steps':0,'vla_steps':0,'tool_steps':0,
        'tools_available':['vla','sam3','graspgen','move_ee','gripper','hold'],
        'motion_planner':'native_whole_body_differential_ik','recorder':'wall_clock_from_first_action'}
    start=time.monotonic();video=TimedVideo(run/'rollout_realtime.mp4',log) if args.video else None
    first_action=None
    site=robot.eef_site_id['right']
    hand_id=raw.sim.model.body_name2id('robot0_right_hand')

    def snapshot():
        data=raw.sim.data
        hand=np.eye(4);hand[:3,:3]=data.body_xmat[hand_id].reshape(3,3);hand[:3,3]=data.body_xpos[hand_id]
        ee=np.eye(4);ee[:3,:3]=data.site_xmat[site].reshape(3,3);ee[:3,3]=data.site_xpos[site]
        # The gym wrapper already vertically flips RGB and depth consistently.
        depth_norm=np.squeeze(obs['video.robot0_agentview_left_depth'])
        near=float(raw.sim.model.vis.map.znear*raw.sim.model.stat.extent)
        far=float(raw.sim.model.vis.map.zfar*raw.sim.model.stat.extent)
        depth=(near/(1-np.clip(depth_norm,0,.999999)*(1-near/far))).astype(np.float32)
        h,w=depth.shape
        state=np.concatenate([obs['state.end_effector_position_relative'],obs['state.end_effector_rotation_relative'],
            obs['state.base_position'],obs['state.base_rotation'],obs['state.gripper_qpos']]).astype(np.float32)
        _,guide=exchange.read()
        views={}
        for label,cam in [('left','robot0_agentview_left'),('right','robot0_agentview_right'),('wrist','robot0_eye_in_hand')]:
            d=np.squeeze(obs['video.'+cam+'_depth'])
            dm=(near/(1-np.clip(d,0,.999999)*(1-near/far))).astype(np.float32)
            views[label]={'rgb':obs['video.'+cam],'depth':dm,
                'K':get_camera_intrinsic_matrix(raw.sim,cam,h,w),
                'T_camera':get_camera_extrinsic_matrix(raw.sim,cam)}
        return {'step':report['steps'],'revision':guide['revision'],'vla_applied':report['vla_steps'],
            'wall_s':max(0.,observed_at-start),'captured_at_monotonic':observed_at,
            'left':obs['video.robot0_agentview_left'],'right':obs['video.robot0_agentview_right'],
            'wrist':obs['video.robot0_eye_in_hand'],'state':state,'depth':depth,
            'K':get_camera_intrinsic_matrix(raw.sim,'robot0_agentview_left',h,w),
            'T_camera':get_camera_extrinsic_matrix(raw.sim,'robot0_agentview_left'),
            'views':views,'ee_pos':ee[:3,3].copy(),'ee_rot':ee[:3,:3].copy(),
            'gripper_command':execution.last_close,
            'T_hand_to_site':np.linalg.inv(hand)@ee,
            'controller_origin_rot':c.origin_ori.copy(),
            'controller_scale':((c.output_max-c.output_min)/(c.input_max-c.input_min)).copy(),
            'kinematics':kinematics.capture(),
            'feedback':executor_feedback(guide,execution,obs['state.gripper_qpos'],ee[:3,3],hz)}

    def wall_state(phase):
        value={'phase':phase,'steps':report['steps'],'limit_s':limit,'elapsed_s':time.monotonic()-start,
            'start_monotonic':start,'deadline_monotonic':start+limit}
        path=run/'wall_state.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)

    threads=[threading.Thread(target=policy_worker,args=(exchange,jobs,results,log),daemon=True,name='vla-worker'),
        threading.Thread(target=tool_worker,args=(exchange,tools,results,log),daemon=True,name='tool-worker'),
        threading.Thread(target=supervise,args=(exchange,instruction,log),daemon=True,name='vlm-supervisor')]
    current=snapshot();exchange.publish(current)
    old_handler=signal.getsignal(signal.SIGALRM)
    class BudgetExpired(Exception): pass
    def expire(signum,frame): raise BudgetExpired()
    signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,limit)
    wall_state('running')
    for thread in threads: thread.start()
    next_tick=time.monotonic()
    log.emit('episode_begin',task=args.task,horizon=horizon,wall_limit_s=limit)
    try:
        while report['steps']<horizon and time.monotonic()-start<limit:
            commands=exchange.take_commands()
            for command in commands:
                applied=execution.apply(command,report['steps'])
                if applied=='stop':
                    report['stopped_by_vlm']=True
                    break
                if isinstance(applied,dict) and applied.get('query'):
                    tools.put_nowait((applied,current,execution.grasp_pose))
                elif isinstance(applied,dict) and applied['tool'] in MOTION_TOOLS:
                    if execution.tool_pending:
                        # A cancelled GPU request may finish, but its result is
                        # version-checked. The new request starts once it clears.
                        execution.phase='waiting_for_cancelled_inference'
            if report.get('stopped_by_vlm'): break
            _,guide=exchange.read()
            while True:
                try: kind,result_guide,source,value,error=results.get_nowait()
                except queue.Empty: break
                if kind=='vla': execution.vla_pending=False
                else: execution.tool_pending=False
                if kind=='query':
                    execution.query_result(result_guide,source,value,error,report['steps'])
                    continue
                if not execution.accepts(result_guide['revision'],kind,report['steps']):
                    continue
                if error:
                    execution.phase='failed_'+error[:150]
                    execution.recent_events.append({'executor':result_guide['tool'],
                        'args':result_guide['args'],'status':'failed','error':error,'step':report['steps']})
                    log.emit('executor_failed',kind=kind,error=error,step=report['steps'])
                    continue
                if kind=='vla':
                    # The robot moved while inference ran. Align the returned
                    # open-loop timeline; this is not a trained RTC policy.
                    # Holds do not consume predicted action time. Count only
                    # VLA actions actually applied while inference was running.
                    age=report['vla_steps']-source['vla_applied']
                    if time.monotonic()-start-source['wall_s']>1.5 or age>=len(value):
                        log.emit('result_discarded',kind=kind,reason='expired_observation',step=report['steps'])
                        continue
                    execution.actions.clear()
                    execution.actions.extend(value[age:age+args.action_window])
                    execution.phase='executing'
                    with (run/'policy_chunks.jsonl').open('a') as f:
                        f.write(json.dumps({'step':report['steps'],'observed_step':source['step'],
                            'revision':guide['revision'],'skipped_prefix':age,'instruction':guide['instruction'],
                            'actions':value.tolist()})+'\n')
                else:
                    execution.motion=MotionExecutor(value)
                    execution.phase=execution.motion.phase
                    log.emit('tool_motion_begin',tool=value.name,revision=guide['revision'],step=report['steps'],evidence=value.evidence)
            # Refresh robot/feedback state after command and result application.
            current=snapshot();exchange.publish(current)
            for command in commands:exchange.acknowledge(command.get('command_id',0))
            if guide['tool']=='vla' and len(execution.actions)<=2 and not execution.vla_pending:
                jobs.put_nowait((guide,current));execution.vla_pending=True
            if guide['tool'] in MOTION_TOOLS and execution.motion is None and not execution.tool_pending and execution.phase in {'preparing','waiting_for_cancelled_inference'}:
                tools.put_nowait((guide,current,execution.grasp_pose));execution.tool_pending=True
            action=execution.hold();source='hold'
            if guide['tool']=='vla' and execution.actions:
                action=execution.actions.popleft();source='vla'
            elif guide['tool'] in MOTION_TOOLS and execution.motion is not None:
                motion_action=execution.motion.tick(current)
                execution.phase=execution.motion.phase
                if motion_action is not None: action=motion_action;source='tool'
                elif execution.motion.status!='moving':
                    if not getattr(execution.motion,'reported',False):
                        execution.motion.reported=True
                        event={'executor':guide['tool'],'args':guide['args'],
                               'status':execution.phase,'step':report['steps']}
                        execution.recent_events.append(event)
                        log.emit('tool_motion_end',revision=guide['revision'],tracking_error=execution.motion.tracking_error,**event)
            execution.last_close=float(action[6])
            if video:
                video.update(current['left'],{'step':report['steps'],'revision':guide['revision'],
                    'executor':source,'phase':execution.phase})
            if first_action is None and (np.linalg.norm(action[:6])>1e-7 or action[6]!=0):
                first_action=time.monotonic()
                log.emit('first_action',step=report['steps'])
                if video: video.start(first_action)
            tick=time.monotonic()
            obs,reward,done,truncated,info=env.step(convert_action(action))
            observed_at=time.monotonic()
            report['env_step_s']+=time.monotonic()-tick
            report['steps']+=1;report[source+'_steps']+=1
            report['success']=bool(info.get('success',False))
            log.emit('control_step',step=report['steps'],revision=guide['revision'],executor=source,
                phase=execution.phase,action=action.tolist(),ee_world=current['ee_pos'].tolist(),env_step_s=time.monotonic()-tick)
            wall_state('running')
            if report['success']: break
            next_tick=max(next_tick+period,time.monotonic())
            delay=next_tick-time.monotonic()
            if delay>0: time.sleep(delay)
        if not report['success'] and time.monotonic()-start>=limit: report['wall_timeout']=True
    except BudgetExpired:
        report['wall_timeout']=True
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        log.emit('episode_error',error=report['error'])
        raise
    finally:
        ended=time.monotonic()
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        exchange.stop.set()
        with exchange.condition: exchange.condition.notify_all()
        report.update(elapsed_s=ended-start,first_action_delay_s=None if first_action is None else first_action-start,
            stale_results_discarded=execution.stale_results,commands_applied=execution.applied_commands)
        wall_state('wall_timeout' if report['wall_timeout'] else 'finished')
        if video:
            video.update(obs['video.robot0_agentview_left'],{'step':report['steps'],'revision':exchange.guide['revision'],
                'executor':'stopped','phase':'finished'})
            report['video']=video.close(ended)
        for label,key in [('agentview','left'),('right','right'),('wrist','wrist')]:
            imageio.imwrite(run/f'final_{label}.jpg',obs[f'video.robot0_{"agentview_"+key if key!="wrist" else "eye_in_hand"}'])
        (run/'result.json').write_text(json.dumps(report,indent=2))
        log.emit('episode_end',result=report)
        print('RESULT',json.dumps(report),flush=True)
        env.close()


if __name__=='__main__': main()
