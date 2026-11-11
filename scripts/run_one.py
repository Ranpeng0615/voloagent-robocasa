"""One native RoboCasa episode through VoLoAgent, with a wall-clock budget."""
import argparse,collections,importlib.util,json,os,signal,time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--task',default='PickPlaceCounterToStove')
parser.add_argument('--split',default='pretrain',choices=['pretrain','target'])
parser.add_argument('--seed',default=7,type=int)
parser.add_argument('--port',default=8001,type=int)
parser.add_argument('--replan-steps',default=5,type=int)
parser.add_argument('--video',action='store_true')
parser.add_argument('--depth',action='store_true')
args=parser.parse_args()
root=Path(os.environ['ROBOCASA_ROOT']);run=Path(os.environ['RUN_DIR']);run.mkdir(parents=True,exist_ok=True)
import gymnasium as gym
import robocasa
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from openpi_client import image_tools,websocket_client_policy

path=root/'repos/VoLoAgent/examples/robocasa/robocasa_eval_client.py'
spec=importlib.util.spec_from_file_location('volo_robocasa_client',path)
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
np.random.seed(args.seed)
env=gym.make(f'robocasa/{args.task}',split=args.split,seed=args.seed,camera_depths=args.depth)
obs,info=env.reset()
raw=env.unwrapped.env
hz=float(raw.control_freq)
horizon=int(get_task_horizon(args.task));limit=horizon/hz+20
client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',args.port)
task_lang=obs['annotation.human.task_description']
report={'task':args.task,'split':args.split,'seed':args.seed,'instruction':task_lang,
        'horizon':horizon,'control_hz':hz,'wall_limit_s':limit,'success':False,
        'wall_timeout':False,'steps':0,'policy_request_s':0.,'env_step_s':0.,
        'checkpoint':'pi05_pretrain_human300','online_gt_state':False}
def pack(obs):
    wire=helpers.build_wire_obs(obs,task_lang,224,env=env if args.depth else None,
                                send_gt_state=False,enable_depth=args.depth)
    right=np.ascontiguousarray(obs['video.robot0_agentview_right'])
    wire['observation/right_image']=image_tools.convert_to_uint8(image_tools.resize_with_pad(right,224,224))
    wire['__episode_id']=f'{args.task}_{args.seed}_0';wire['__step']=report['steps']
    return wire
class BudgetExpired(Exception):pass
def expire(signum,frame):raise BudgetExpired('wall budget exhausted')
start=time.monotonic();actions=collections.deque();frames=[]
def write_state(phase):
    state={'phase':phase,'task':args.task,'steps':report['steps'],'limit_s':limit,
           'elapsed_s':time.monotonic()-start,'start_monotonic':start,'deadline_monotonic':start+limit}
    p=run/'wall_state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state));tmp.replace(p)
print('EPISODE',json.dumps(report),flush=True)
old=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,limit)
write_state('running')
try:
    while report['steps']<horizon:
        if not actions:
            wire=pack(obs);tick=time.monotonic()
            try:response=client.infer(wire)
            finally:report['policy_request_s']+=time.monotonic()-tick
            chunk=np.asarray(response['actions'])
            if not np.isfinite(chunk).all():raise ValueError('nonfinite policy action')
            trace={'step':report['steps'],'elapsed_s':time.monotonic()-start,
                   'instruction':response.get('orchestrator_instruction',task_lang),
                   'actions':chunk.tolist()}
            with (run/'policy_chunks.jsonl').open('a') as f:f.write(json.dumps(trace)+'\n')
            for key in ['orchestrator_subgoals','orchestrator_episode_log_dir']:
                if key in response:report[key]=response[key]
            actions.extend(chunk[:args.replan_steps])
        if args.video:frames.append(np.ascontiguousarray(obs['video.robot0_agentview_left']).copy())
        action=np.asarray(actions.popleft())
        if len(action)<12:action=np.pad(action,(0,12-len(action)))
        tick=time.monotonic()
        obs,reward,done,truncated,info=env.step(convert_action(action))
        report['env_step_s']+=time.monotonic()-tick
        report['steps']+=1;report['success']=bool(info.get('success',False))
        write_state('running')
        if report['success']:break
except BudgetExpired:report['wall_timeout']=True
except Exception as exc:
    report['error']=f'{type(exc).__name__}: {exc}'
    raise
finally:
    signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    report['elapsed_s']=time.monotonic()-start
    write_state('wall_timeout' if report['wall_timeout'] else 'finished')
    for key,label in [('video.robot0_agentview_left','agentview'),('video.robot0_agentview_right','right'),('video.robot0_eye_in_hand','wrist')]:
        imageio.imwrite(run/f'final_{label}.jpg',np.asarray(obs[key]))
    (run/'result.json').write_text(json.dumps(report,indent=2))
    print('RESULT',json.dumps(report),flush=True)
    if args.video and frames:imageio.mimwrite(run/'rollout.mp4',frames,fps=hz)
    env.close()
