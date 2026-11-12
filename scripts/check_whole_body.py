"""Native headless motion check; no VLM/VLA services or grasp success claim.

The first check removes arm/torso freedom from the IK allocation so translation
must come from the base. The second checks normal combined tracking. This is
an isolated simulator instance, not a scored task attempt.
"""
import json,random,os
from pathlib import Path
import numpy as np
import robocasa,robosuite
from robosuite.controllers import load_composite_controller_config
from whole_body import RobotKinematics
from robocasa_tools import MotionExecutor,make_plan
random.seed(7);np.random.seed(7)
env=robosuite.make(env_name='SteamInMicrowave',robots='PandaOmron',controller_configs=load_composite_controller_config(robot='PandaOmron'),seed=7,layout_ids=-2,style_ids=-2,obj_instance_split='pretrain',has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,camera_names=[],use_object_obs=False,ignore_done=True)
env.hard_reset=False;env.reset();robot=env.robots[0];sim=env.sim;parts=robot.composite_controller.part_controllers;c=parts['right']
kin=RobotKinematics(robot,env.control_freq)
site=robot.eef_site_id['right']
def snapshot():
 return {'ee_pos':sim.data.site_xpos[site].copy(),'ee_rot':sim.data.site_xmat[site].reshape(3,3).copy(),
  'gripper_command':0.,'controller_origin_rot':c.origin_ori.copy(),'controller_scale':(c.output_max-c.output_min)/(c.input_max-c.input_min),'kinematics':kin.capture()}
def step(a):
 action=robot.composite_controller.create_action_vector({'right':a[:6],'right_gripper':np.array([-1. if a[6]<.5 else 1.]),'base':a[7:10],'torso':a[10:11],'base_mode':-1 if a[11]<.5 else 1})
 env.step(action)


# Force the IK allocator to use the physical mobile base, not arm reach.
snap=snapshot();base_goal=snap['ee_pos']+c.origin_ori@np.array([-.08,0,0]);base_anchor=snap['kinematics']['q'][:3].copy()
base_motion=MotionExecutor(make_plan({'tool':'move_ee','revision':1,'args':{'position_world':base_goal.tolist(),'timeout_s':20}},snap,type('Log',(),{'emit':lambda *a,**k:None})()))
for base_step in range(240):
 snap=snapshot();k=snap['kinematics'];k['limits'][2:]=np.column_stack([k['q'][2:],k['q'][2:]])
 a=base_motion.tick(snap)
 if a is None:break
 step(a)
base_result={'status':base_motion.status,'steps':base_step,'error_m':float(np.linalg.norm(base_goal-snapshot()['ee_pos'])),'base_displacement_m':float(np.linalg.norm(kin.capture()['q'][:2]-base_anchor[:2]))}
print('BASE_ONLY',json.dumps(base_result),flush=True)
snap=snapshot();goal=snap['ee_pos']+c.origin_ori@np.array([.12,0,.04]);anchor=snap['kinematics']['q'][:3].copy()
motion=MotionExecutor(make_plan({'tool':'move_ee','revision':2,'args':{'position_world':goal.tolist(),'timeout_s':20}},snap,type('Log',(),{'emit':lambda *a,**k:None})()))
records=[]
for i in range(200):
 snap=snapshot();a=motion.tick(snap)
 if a is None:break
 step(a)
 if i%20==0:
  row={'step':i,'error_m':float(np.linalg.norm(goal-snapshot()['ee_pos'])),'base_q':kin.capture()['q'][:3].tolist(),'torso_q':float(kin.capture()['q'][3]),'action':a.tolist()};records.append(row);print('TRACK',json.dumps(row),flush=True)
result={'base_only':base_result,'status':motion.status,'steps':i,'position_error_m':float(np.linalg.norm(goal-snapshot()['ee_pos'])),'base_displacement_m':float(np.linalg.norm(kin.capture()['q'][:2]-anchor[:2])),'records':records}
result['passed']=(base_result['status']=='motion_done_unverified' and base_result['base_displacement_m']>.04 and base_result['error_m']<.009 and motion.status=='motion_done_unverified')
path=Path(os.environ['ROBOCASA_ROOT'])/'cache/whole_body_smoke.json';path.write_text(json.dumps(result,indent=2))
print('SMOKE',json.dumps(result),flush=True)
env.close()
if not result['passed']:raise SystemExit('Native whole-body motion check failed')
