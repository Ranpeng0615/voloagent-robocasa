"""Continuous high-level visual supervision using the configured DeepSeek bridge."""
import base64
import io
import json
import time
from PIL import Image
import httpx

SYSTEM = '''You supervise a robot in RoboCasa. A separate executor CONTINUES moving
while you reason. Each request contains a snapshot taken at observed_step; it is
not a live video. Use image evidence and executor feedback, not assumed action
success. A completed grasp motion is NOT proof of holding the object. If an
object remains on its support after a failed grasp, REFINE the goal back to
grasping instead of allowing a transport action to continue. Use a short concrete
instruction appropriate to the CURRENT stage. Keep the user's complete task in
memory. You may decompose it into a few subgoals and track them in memory.

Return ONLY JSON:
{"command":"continue|refine|switch_tool|hold|stop",
 "tool":"vla|grasp|place|hold", "instruction":"current concrete instruction",
 "args":{}, "reason":"brief visible evidence", "memory":"compact task progress"}.

Tools:
- vla: official RoboCasa pi05, accepts a language instruction and all 3 cameras.
  Use for general movements, appliance/door interactions and tasks not covered
  by geometric tools.
- grasp: author's SAM3 text segmentation + RGB-D point cloud + GraspGen
  prediction/collision filtering, then RoboCasa closed-loop approach, close,
  lift. args={"target":"short visible object name"}. Suitable for a visible,
  reachable loose object when geometric grasping is likely to help. It reports
  motion_done_unverified after lifting;
  YOU verify visually that the object is held. It does not transport it.
- place: SAM3 + depth localize a visible support; closed-loop transport,
  lower, open and retreat. args={"destination":"short visible support name"}.
  Use only after visual evidence that a grasp succeeded. Limited to reachable
  supports; use vla for navigation/articulated or obstructed placements.
- hold: retain position and last commanded gripper closure.

continue retains the active execution and instruction. During a normal active
grasp/place motion, continue unless there is clear failure or wrong target.
At awaiting_initial_guidance the robot is holding still; select a tool with
switch_tool to begin. The task timer is already running.
refine changes the current VLA instruction and invalidates its pending actions.
switch_tool cancels the current executor and starts the selected tool. Changing
from grasp to place (or changing object) requires switch_tool. Do not repeatedly
restart a tool that is still preparing or moving. After motion_done_unverified,
verify and choose the next tool; do not continue a finished tool indefinitely.
hold suspends motion. stop ends the episode only if the task is visibly complete
or cannot continue; a simulator success check remains the evaluation authority.
No simulator object coordinates, segmentation labels or success flags are given.

SELECT EXECUTORS DYNAMICALLY. There is no fixed VLA/tool schedule, and picking
does not automatically require a grasp tool. Choose whichever executor is most
likely to advance the current stage. Keep VLA when it is making visible progress.
If it closes on empty space, leaves an object behind while starting transport,
stalls, or repeats a failed motion, actively consider grasp/place tools instead
of repeatedly rewording the same instruction. Use SAM3/GraspGen only when the
object is visible and the reachable geometric approach is appropriate. If a
tool cannot detect the target, cannot converge, or cannot handle an articulated
interaction, switch back to VLA with the specific unresolved goal. Do not force
every available tool to be used. Explain each switch using observed evidence.
Compare the previous external image, when supplied, with the current images;
robot motion alone is not proof that the object or task made progress.
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
    if value.get('command') not in {'continue','refine','switch_tool','hold','stop'}:
        raise ValueError('Invalid command')
    if value.get('tool') not in {'vla','grasp','place','hold'}:
        raise ValueError('Invalid tool')
    if not isinstance(value.get('instruction',''),str) or not isinstance(value.get('args',{}),dict):
        raise ValueError('Invalid instruction/args')
    if value['command'] == 'refine' and value['tool'] != 'vla':
        raise ValueError('refine is for VLA; use switch_tool to restart geometric tools')
    if value['command'] == 'switch_tool':
        field = {'grasp':'target','place':'destination'}.get(value['tool'])
        if field and not isinstance(value.get('args',{}).get(field),str):
            raise ValueError('Missing tool argument')
    return value


def supervise(exchange, task, log):
    memory = ''
    previous_step = -1
    previous_view = None
    with httpx.Client(timeout=httpx.Timeout(60, connect=5), trust_env=False) as client:
        while not exchange.stop.is_set():
            snapshot, guide = exchange.read()
            if snapshot is None or snapshot['step'] == previous_step:
                exchange.stop.wait(.03)
                continue
            previous_step = snapshot['step']
            feedback = {k:snapshot[k] for k in ('step','wall_s','feedback')}
            text = json.dumps({'task':task, 'observed_step':snapshot['step'],
                'active':guide, 'feedback':feedback, 'memory':memory},ensure_ascii=False)
            content = [{'type':'text','text':text}]
            for name, label in [('left','Left external'),('right','Right external'),('wrist','Wrist')]:
                content.extend(image_part(snapshot[name], label))
            if previous_view is not None:
                content.extend(image_part(previous_view[1],f'Previous left external at step {previous_view[0]} (for progress comparison)'))
            tick = time.monotonic()
            log.emit('vlm_begin', observed_step=snapshot['step'], revision=guide['revision'])
            try:
                response = client.post('http://127.0.0.1:8111/v1/chat/completions',json={
                    'model':'deepseek-flash','messages':[{'role':'system','content':SYSTEM},
                    {'role':'user','content':content}], 'response_format':{'type':'json_object'}})
                response.raise_for_status()
                raw = response.json()['choices'][0]['message']['content']
                decision = parse_decision(raw)
                current, _ = exchange.read()
                log.emit('vlm_end', observed_step=snapshot['step'], current_step=current['step'],
                    revision=guide['revision'], duration_s=time.monotonic()-tick, decision=decision)
                if exchange.stop.is_set():
                    break
                memory = decision.get('memory',memory)
                previous_view = (snapshot['step'],snapshot['left'])
                exchange.command(decision,guide['revision'],snapshot['step'])
                # Wait for the execution loop to apply this decision before
                # selecting the next latest frame. Never queue old snapshots.
                with exchange.condition:
                    exchange.condition.wait_for(lambda: not exchange.commands or exchange.stop.is_set(), timeout=.2)
            except Exception as exc:
                log.emit('vlm_error', error=f'{type(exc).__name__}: {exc}',duration_s=time.monotonic()-tick)
                exchange.stop.wait(.3)
