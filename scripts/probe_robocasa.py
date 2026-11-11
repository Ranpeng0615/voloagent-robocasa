"""Inspect the native robot/camera contract without moving the robot."""
import json, os
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import gymnasium as gym
import robocasa
from robocasa.utils.dataset_registry_utils import get_task_horizon
root = Path(os.environ['ROBOCASA_ROOT'])
env = gym.make('robocasa/BreadAndCheese', split='pretrain', seed=7, camera_depths=True)
obs, info = env.reset()
raw = env.unwrapped.env
robot = raw.robots[0]
c = robot.composite_controller.part_controllers['right']
data = {'instruction': obs['annotation.human.task_description'], 'horizon': get_task_horizon('BreadAndCheese'),
        'hz': raw.control_freq, 'obs_shapes': {k: np.shape(v) for k,v in obs.items()},
        'eef_site_id': robot.eef_site_id, 'controller': {k: str(getattr(c,k,None)) for k in
        ['input_type','input_ref_frame','output_min','output_max','input_min','input_max','ref_name','origin_pos','origin_ori','_goal_update_mode']}}
data['gripper_bodies'] = [n for n in raw.sim.model.body_names if 'gripper' in n or 'hand' in n]
data['gripper_sites'] = [n for n in raw.sim.model.site_names if 'grip' in n]
data['site_local'] = {n: raw.sim.model.site_pos[raw.sim.model.site_name2id(n)].tolist() for n in data['gripper_sites']}
(root/'cache/robocasa-contract.json').write_text(json.dumps(data,indent=2))
imageio.imwrite(root/'cache/bread-and-cheese-initial.jpg', obs['video.robot0_agentview_left'])
print(json.dumps(data,indent=2))
env.close()
