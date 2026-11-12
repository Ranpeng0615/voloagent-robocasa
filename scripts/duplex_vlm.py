"""Continuous high-level visual supervision using the configured DeepSeek bridge."""
import base64
import io
import json
import time
from PIL import Image
import httpx
from duplex_contract import validate_decision

SYSTEM = '''You are the high-level visual controller for a RoboCasa robot.
VLA starts on the full task and runs while you reason. Use the latest camera
images, task memory and actual tool results to decide what to do next. Keep VLA
when it works; use other tools when they can resolve a problem. You choose the
subgoals, tool order, motion targets, gripper actions and recovery. No fixed grasp
or placement sequence is inserted by the executor.

Return JSON:
{"command":"continue|refine|switch_tool|call_tool|hold|stop",
 "tool":"vla|sam3|graspgen|move_ee|gripper|hold",
 "instruction":"short action goal, or empty when continuing", "args":{}, "reason":"visible evidence",
 "memory":"compact task progress", "plan":["initial task plan"]}.
On initial_plan, establish a plan. Later calls include that plan and memory;
update progress and any plan adjustments in memory. Each request contains fixed
images, not live input; the robot can move while a response is generated.

VLA instruction is passed verbatim to the robot policy. Write a short imperative
for the current action goal, such as "Place the bowl in the microwave."
Put scene descriptions, grasp assessments, explanations and future plans in
reason/memory, not in instruction. Choose the goal from the actual current
images; if an object is already held, describe the remaining action rather than
repeating acquisition. Keep the same instruction while pursuing the same goal.
For continue, omit instruction or return instruction=""; the active instruction
is retained. Use refine when you intentionally change the goal or corrective
action, rather than paraphrasing the same goal on every observation. You still
choose when to change goals, recover, hold or use tools.

Tools and arguments:
- vla: language-guided robot policy using all three cameras. refine updates its
  instruction. switch_tool selects it after another motion tool.
- sam3: call_tool with {"target":"visible object or support name"}. Returns a
  calibrated RGB-D position estimate and segmentation evidence, without motion.
- graspgen: call_tool with {"target":"visible object name"}. Runs author's SAM3
  and GraspGen and returns an end-effector grasp pose in world coordinates, pose_ref,
  approach_axis_world, confidence and observation time. This predicts a pose;
  it does not move, open, close, lift or declare grasp success. Optional
  topdown_only=true requests top-down filtering; default is false.
- move_ee: switch_tool with one requested end-effector pose. Use
  {"pose_ref":"grasp:<returned id>"} or {"position_world":[x,y,z]} or
  {"pose_ref":"current","offset_world":[dx,dy,dz]}.
  Optional quaternion_xyzw sets orientation; otherwise position-only moves keep
  current orientation, and grasp references use the returned grasp orientation.
  Optional offset_world adds a translation in world axes; offset_local adds one
  in the selected pose axes. All distances are meters. Gripper is preserved
  unless args.gripper is explicitly open or close. Local whole-body IK uses the
  mobile base, torso and arm. No intermediate waypoint is added. There is no
  per-motion timeout unless you set timeout_s. Feedback reports tracking errors;
  reaching the numerical pose is motion_done_unverified, not object possession.
- gripper: switch_tool with {"command":"open|close"}; optional duration_s
  defaults to 0.5 seconds. Operates only the gripper while holding robot position.
- hold: retain position and the last commanded gripper state.

call_tool launches perception asynchronously and preserves the current VLA or
motion execution. Only one tool preparation/query is pending at once; read
feedback.tool_pending and latest_tool_result. Tool results are tagged with their
source observation time. feedback.grasp_pose retains the latest successful grasp
prediction and its pose_ref, even after another query. Neither query results nor
failure reports automatically change the executor. Decide whether to hold for
perception or continue moving, and whether a predicted pose still fits the images.

continue keeps the active motion and instruction when instruction is empty.
An explicitly changed VLA instruction still applies as refine;
changed motion tool/args apply as switch_tool. A switch or refine cancels old
motion and queued actions; late motion results cannot restart it. To retry a
finished motion, use switch_tool. Commands and required arguments are validated;
progress assessments do not force tool selection or hold.

Assess grasp success from images and object motion. Gripper commands, raw finger
spacing and a completed tool motion are not proof of holding or failure. A held
object can keep the fingers separated. Compare the previous external image with
current views when useful. Buffer length is not task progress. stop ends the
trial; the native simulator check determines benchmark success.
'''


def image_part(array, label):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format='JPEG', quality=85)
    return [{'type': 'text', 'text': label}, {'type': 'image_url', 'image_url': {
        'url': 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()}}]


def parse_decision(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n',1)[1].rsplit('```',1)[0]
    value = json.loads(text)
    return validate_decision(value)


def supervise(exchange, task, log):
    memory = ''
    initial_plan = None
    previous_step = -1
    previous_view = None
    with httpx.Client(timeout=httpx.Timeout(60, connect=5), trust_env=False) as client:
        while not exchange.stop.is_set():
            snapshot, guide = exchange.read()
            if snapshot is None or snapshot['step'] == previous_step:
                exchange.stop.wait(.03)
                continue
            previous_step = snapshot['step']
            planning=initial_plan is None
            request_kind='initial_plan' if planning else 'monitor'
            text = json.dumps({'task':task, 'observed_step':snapshot['step'],
                'observed_wall_s':snapshot['wall_s'],
                'frame_age_at_pack_s':time.monotonic()-snapshot['captured_at_monotonic'],
                'request_kind':request_kind, 'initial_plan':initial_plan,
                'active':guide, 'feedback':snapshot['feedback'], 'memory':memory},ensure_ascii=False)
            content = [{'type':'text','text':text}]
            for name, label in [('left','Left external'),('right','Right external'),('wrist','Wrist')]:
                content.extend(image_part(snapshot[name], label))
            if previous_view is not None:
                content.extend(image_part(previous_view[1],f'Previous left external at step {previous_view[0]} (for progress comparison)'))
            tick = time.monotonic()
            log.emit('vlm_begin', observed_step=snapshot['step'], revision=guide['revision'],
                request_kind=request_kind,thinking='max' if planning else 'disabled',
                observed_wall_s=snapshot['wall_s'],captured_at_monotonic=snapshot['captured_at_monotonic'],
                frame_age_s=tick-snapshot['captured_at_monotonic'])
            try:
                payload={
                    'model':'deepseek-flash','messages':[{'role':'system','content':SYSTEM},
                    {'role':'user','content':content}], 'response_format':{'type':'json_object'},
                    'thinking':{'type':'enabled' if planning else 'disabled'},
                    'reasoning_effort':'max' if planning else 'none'}
                if not planning:payload['max_tokens']=1024
                response = client.post('http://127.0.0.1:8111/v1/chat/completions',json=payload)
                response.raise_for_status()
                raw = response.json()['choices'][0]['message']['content']
                decision = parse_decision(raw)
                if planning:
                    plan=decision.get('plan')
                    initial_plan=plan if isinstance(plan,list) and plan and all(isinstance(s,str) and s.strip() for s in plan) else [task]
                current, _ = exchange.read()
                log.emit('vlm_end', observed_step=snapshot['step'], current_step=current['step'],
                    revision=guide['revision'], duration_s=time.monotonic()-tick, decision=decision,
                    request_kind=request_kind,thinking='max' if planning else 'disabled',
                    input_frame_age_s=time.monotonic()-snapshot['captured_at_monotonic'],
                    latest_frame_age_s=time.monotonic()-current['captured_at_monotonic'],
                    current_phase=current['feedback']['phase'])
                if exchange.stop.is_set():
                    break
                memory = decision.get('memory',memory)
                previous_view = (snapshot['step'],snapshot['left'])
                command_id=exchange.command(decision,guide['revision'],snapshot['step'])
                # Wait for the execution loop to apply this decision before
                # selecting the next latest frame. Never queue old snapshots.
                with exchange.condition:
                    exchange.condition.wait_for(lambda: exchange.acknowledged>=command_id or exchange.stop.is_set())
            except Exception as exc:
                log.emit('vlm_error', error=f'{type(exc).__name__}: {exc}',duration_s=time.monotonic()-tick)
                exchange.stop.wait(.3)
