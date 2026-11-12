"""Continuous high-level visual supervision using the configured DeepSeek bridge."""
import base64
import io
import json
import time
from PIL import Image
import httpx
from duplex_contract import validate_decision

SYSTEM = '''You supervise a robot in RoboCasa. A separate executor CONTINUES moving
while you reason. Each request contains a snapshot taken at observed_step; it is
not a live video. Use image evidence and executor feedback, not assumed action
success. A completed grasp motion is NOT proof of holding the object. If an
object remains on its support after a failed VLA grasp, return to acquisition
and select grasp recovery instead of continuing transport. Use a short concrete
instruction appropriate to the CURRENT stage. Keep the user's complete task in
memory. You may decompose it into a few subgoals and track them in memory.

Return ONLY JSON:
{"command":"continue|refine|switch_tool|hold|stop",
 "tool":"vla|grasp|place|hold", "instruction":"current concrete instruction",
 "args":{}, "target":"short name of current object", "progress":"normal|uncertain|stalled|failed",
 "stage":"acquire|transport|place|other",
 "grasp_status":"unverified|confirmed|failed|lost",
 "reason":"brief visible evidence", "memory":"compact task progress",
 "plan":["ordered subgoal; initial_plan request only"]}.

For request_kind=initial_plan, establish a short ordered plan for the full task
and return it in plan, but guide only the current unresolved stage. Later requests
contain that saved initial_plan: do not rebuild it or repeat it in the output.
For request_kind=monitor, directly assess current images and output a short
decision. Keep reason under 30 words and memory under 100 words. A failed grasp
must remain unresolved in memory until visual evidence establishes recovery.

VLA IS THE DEFAULT EXECUTOR and starts immediately with the native task
instruction. Your first request does not block it. At step 0 there is no motion
history: normally continue VLA, do not choose geometric grasp just because an
object is visible. When you observe an anomaly or insufficient progress, consider
grasp/place instead of repeatedly rewording a failed VLA instruction. You choose
the executor and recovery sequence from images, recent outcomes and tool limits.
The runtime does not force a grasp, a hold, or a VLA fallback from your progress,
stage or grasp_status fields. Report those fields honestly; a place call after a
successful tool grasp does not require reporting a new failure. Always name the
current object in target and keep that name stable across retries. Use recent
executor events to decide whether a tool retry or VLA recovery is appropriate.

For a multi-stage task, track unresolved stages and select useful instruction
boundaries yourself. You can separate acquisition from transport when it helps
resolve a failed grasp. Use images and the object's motion relative to the
gripper to assess holding; a close command or finger spacing alone is not proof.
If the object remains behind, correct the acquisition rather than continuing
transport. Do not repeatedly pause for a ritual verification lift when the images
already establish holding. Decide whether to continue, refine, hold or switch
tools yourself; the executor does not turn unverified holding into a forced hold.

Tools:
- vla: official RoboCasa pi05, accepts a language instruction and all 3 cameras.
  Use for general movements, appliance/door interactions and tasks not covered
  by geometric tools.
- grasp: author's SAM3 text segmentation + RGB-D point cloud + GraspGen
  prediction/collision filtering, then RoboCasa closed-loop approach, close,
  lift with local whole-body IK (mobile base, torso and arm).
  args={"target":"short visible object name"}. Suitable for a visible,
  reachable loose object when geometric grasping is likely to help. It reports
  motion_done_unverified after lifting;
  YOU verify visually that the object is held. It does not transport it.
  While preparing/approaching, not yet holding is expected: report unverified,
  not a new failed grasp merely because the previous VLA attempt failed.
  This tool opens the gripper during clearance and approach. Calling it while
  already holding an object can release that object; inspect the images first.
- place: SAM3 + depth localize a visible support; closed-loop transport,
  lower, open and retreat. args={"destination":"short visible support name"}.
  It needs the held-object offset from a preceding geometric grasp; it cannot
  directly place an object grasped only by VLA. Use VLA for that placement.
  Use after visual evidence that a grasp succeeded. Limited to reachable
  supports; use vla for navigation/articulated or obstructed placements.
- hold: retain position and last commanded gripper closure.

continue with an unchanged instruction retains active execution. A changed VLA
instruction is applied as refine: old queued actions are cleared and old
inference results are invalidated. A changed tool or geometric target/destination
is applied as switch_tool. For an unchanged geometric tool and args, changed
descriptive text updates context without restarting its motion. During a normal active
grasp/place motion, continue unless there is clear failure or wrong target.
refine changes the current VLA instruction and invalidates its pending actions.
switch_tool cancels the current executor and starts the selected tool. Changing
from grasp to place (or changing object) requires switch_tool. Do not repeatedly
restart a tool that is still preparing or moving. After motion_done_unverified,
verify and choose the next tool; do not continue a finished tool indefinitely.
hold suspends motion. stop ends the episode only if the task is visibly complete
or cannot continue; a simulator success check remains the evaluation authority.
No simulator object coordinates, segmentation labels or success flags are given.

SELECT RECOVERY EXECUTORS DYNAMICALLY. There is no fixed recovery tool schedule.
Picking starts with VLA. Keep VLA when it is making visible progress.
If it closes on empty space, leaves an object behind while starting transport,
stalls, or repeats a failed grasp, prefer an appropriate recovery tool over
repeating the same ineffective VLA instruction. Use SAM3/GraspGen when the
object is visible and the reachable geometric approach is appropriate. If a
tool fails, use its feedback and the images to choose a corrected retry or VLA
with the specific unresolved goal. Do not force
every available tool to be used. Explain each switch using observed evidence.
Compare the previous external image, when supplied, with the current images;
robot motion alone is not proof that the object or task made progress.

Feedback semantics: gripper.commanded is explicitly open or close in the native
interface; it reports the command, not a measured finger state.
gripper.finger_separation_m is a raw distance in meters, with no open/closed or
grasp-success classification. A held object can keep the fingers separated;
spacing alone cannot establish either holding or an empty grasp.
Assess grasp success or failure from images, not a distance threshold. action_buffer
is a rolling control buffer refilled by VLA, NOT a task/subgoal queue. Its length
and draining do not indicate task progress. Do not wait for it to empty to verify
a grasp. gripper_preview lists actual queued commands only. Read phase and
last_guidance_result: failed/completed tools cannot resume with unchanged
continue; choose the next executor or explicitly retry with switch_tool.
Observation timestamps and frame ages describe input freshness. Images are fixed
for this request; the robot may move during your reasoning. A command applied
successfully is NOT evidence of task progress.
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
