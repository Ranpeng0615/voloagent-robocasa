"""Small shared command/feedback contract for the supervisor and executor."""
from __future__ import annotations


def clean_instruction(value):
    return ' '.join(value.split())


def validate_decision(value):
    if not isinstance(value, dict) or value.get('command') not in {'continue','refine','switch_tool','hold','stop'}:
        raise ValueError('Invalid command')
    if value.get('tool') not in {'vla','grasp','place','hold'}:
        raise ValueError('Invalid tool')
    if not isinstance(value.get('instruction',''), str) or not isinstance(value.get('args',{}), dict):
        raise ValueError('Invalid instruction/args')
    if not isinstance(value.get('reason',''),str):
        raise ValueError('Invalid visual evidence')
    if not isinstance(value.get('target',''),str):
        raise ValueError('Invalid recovery target')
    if value.get('progress','uncertain') not in {'normal','uncertain','stalled','failed'}:
        raise ValueError('Invalid progress assessment')
    if value.get('grasp_status','unverified') not in {'unverified','confirmed','failed','lost'}:
        raise ValueError('Invalid grasp assessment')
    if value.get('stage','other') not in {'acquire','transport','place','other'}:
        raise ValueError('Invalid stage')
    if value['command']=='refine' and value['tool']!='vla':
        raise ValueError('refine is for VLA; use switch_tool to restart geometric tools')
    if value['command']=='switch_tool':
        validate_tool_args(value['tool'], value.get('args',{}))
    return value


def validate_tool_args(tool, args):
    field={'grasp':'target','place':'destination'}.get(tool)
    if field and (not isinstance(args.get(field),str) or not args[field].strip()):
        raise ValueError('Missing tool argument: '+field)


def resolve_decision(value, active, phase, failed_tool_targets=()):
    """Resolve continue with changed guidance; never silently discard changes.

    Geometric tools consume args, VLA consumes instruction. Updating only the
    description of an unchanged geometric motion does not restart that motion.
    """
    validate_decision(value)
    result=dict(value)
    kind=value['command']
    tool='hold' if kind=='hold' else value['tool']
    instruction=clean_instruction(value.get('instruction','')) or active['instruction']
    args=value.get('args',{})
    if kind=='continue':
        if tool==active['tool'] and not args:
            args=active['args']
        if tool!=active['tool'] or args!=active['args']:
            kind='switch_tool'
        elif tool=='vla' and instruction!=clean_instruction(active['instruction']):
            kind='refine'
        elif phase.startswith('failed') or phase in {'motion_done_unverified','awaiting_initial_guidance'}:
            raise ValueError('No active motion to continue; verify outcome and select/refine an executor')
    # A declared failed VLA grasp must try another available capability before
    # another language-only retry. Do not infer an object name from prose.
    failed_acquisition=(value.get('stage')=='acquire' and
        (value.get('progress') in {'stalled','failed'} or value.get('grasp_status') in {'failed','lost'}))
    if active['tool'] in {'vla','hold'} and tool in {'vla','hold'} and kind!='stop' and failed_acquisition and value.get('observed_step')!=0:
        target_value=value.get('target') or args.get('target') or ''
        target=clean_instruction(target_value) if isinstance(target_value,str) else ''
        if target and target.casefold() not in failed_tool_targets:
            kind,tool='switch_tool','grasp'
            args={'target':target}
            instruction=f'Grasp {target} and lift slightly for visual verification.'
            result.update(progress='failed',stage='acquire',guard_reason='failed_vla_grasp_try_grasp_tool')
        elif not target:
            kind,tool='hold','hold'
            result['guard_reason']='missing_recovery_target; name the failed object and select grasp'
        else:
            result['guard_reason']='grasp_tool_already_failed_for_target; VLA recovery allowed'
    if kind=='switch_tool':
        validate_tool_args(tool,args)
        if tool in {'grasp','place'}:
            if value.get('observed_step')==0:
                raise ValueError('Initial snapshot has no VLA execution history; observe progress before recovery tools')
            if result.get('progress') not in {'stalled','failed'} or not str(value.get('reason','')).strip():
                raise ValueError('Geometric tools require an observed anomaly or lack of progress')
    # These are guards on the supervisor's declared visual assessment, not a
    # detector of hidden VLA stages or an independent proof of object possession.
    if kind not in {'stop','hold'}:
        holding_confirmed=value.get('grasp_status')=='confirmed' and bool(value.get('reason','').strip())
        if (value.get('stage') in {'transport','place'} or tool=='place') and not holding_confirmed:
            kind,tool='hold','hold'
            result['guard_reason']='holding_unverified; verify grasp before transport/place'
        elif kind=='continue' and (value.get('progress')=='failed' or
                (value.get('grasp_status') in {'failed','lost'} and not
                 (active['tool']=='grasp' and phase in {'preparing','waiting_for_cancelled_inference','clear','above_target','pregrasp','approach'}))):
            kind,tool='hold','hold'
            result['guard_reason']='reported_failure_requires_recovery; unchanged continue blocked'
    result.update(command=kind,tool=tool,instruction=instruction,args=dict(args))
    return result


def gripper_command(raw_value):
    # RoboCasa GymWrapper: action.gripper_close < .5 -> native -1 (open).
    return 'close' if float(raw_value)>=.5 else 'open'


def executor_feedback(guide, execution, qpos, ee_world, control_hz):
    separation=abs(float(qpos[0])-float(qpos[1]))
    measured='open' if separation>=.07 else 'closed' if separation<=.01 else 'between'
    return {
        'schema_version':1,
        'executor':guide['tool'], 'phase':execution.phase,
        'gripper':{'commanded':gripper_command(execution.last_close),
            'measured':measured, 'finger_separation_m':separation,
            'holding_object':'unverified'},
        'action_buffer':{'kind':'rolling_control_buffer', 'queued_steps':len(execution.actions),
            'buffered_sim_s':len(execution.actions)/control_hz,
            'auto_refilled':guide['tool']=='vla', 'indicates_task_progress':False,
            'gripper_preview':[gripper_command(action[6]) for action in execution.actions]},
        'vla_pending':execution.vla_pending, 'tool_pending':execution.tool_pending,
        'robot_ee_world':[float(v) for v in ee_world],
        'tool_motion':None if execution.motion is None else dict(execution.motion.tracking_error),
        'mobile_manipulation_available':True,
        'motion_success_is_unverified':True,
        'last_guidance_result':execution.last_guidance_result,
        'last_visual_assessment':execution.last_visual_assessment,
        'recent_executor_events':list(execution.recent_events)}
