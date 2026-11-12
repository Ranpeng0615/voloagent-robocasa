"""Control invariants that matter when inference and motion overlap."""
import sys
from itertools import product
import json
import threading
import queue
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from duplex_core import Exchange, EventLog, TimedVideo
from duplex_vlm import parse_decision, supervise, progress_reference
from duplex_contract import executor_feedback
from run_duplex import ExecutionState, policy_wire, tool_worker
from robocasa_tools import MotionExecutor, ToolPlan, Waypoint, make_plan, perceive


class DuplexControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bus=Exchange('pick bread')
        self.log=EventLog(Path(self.temp.name)/'events.jsonl')
        self.addCleanup(self.log.file.close)
        self.owner=ExecutionState(self.bus,self.log)

    def command(self, command, tool='vla', revision=0, **extra):
        return {'command':command,'tool':tool,'based_revision':revision,'received_at':time.monotonic(),
                'instruction':'retry grasping bread','args':{},
                'progress':'normal','stage':'acquire','grasp_status':'unverified','target':'bread',
                'reason':'Object remained on its support over repeated observations',**extra}

    def test_refine_flushes_immediately_without_waiting_for_inflight_vla(self):
        self.owner.actions.extend([np.ones(12) for _ in range(5)])
        self.owner.vla_pending=True
        guide=self.owner.apply(self.command('refine'),20)
        self.assertEqual(len(self.owner.actions),0)
        self.assertEqual(guide['revision'],1)
        self.assertTrue(self.owner.vla_pending)
        # A response remains outstanding; the current control action is hold.
        np.testing.assert_array_equal(self.owner.hold()[:6],np.zeros(6))

    def test_stale_high_level_command_cannot_restart_cancelled_tool(self):
        self.owner.apply(self.command('hold',tool='hold'),21)
        stale=self.command('switch_tool',tool='move_ee',args={'pose_ref':'current','offset_world':[0,0,.05]})
        self.assertIsNone(self.owner.apply(stale,22))
        self.assertEqual(self.bus.guide['tool'],'hold')
        self.assertEqual(self.bus.guide['revision'],1)

    def test_hold_preserves_closed_gripper(self):
        self.owner.last_close=1.
        self.owner.apply(self.command('hold',tool='hold'),23)
        self.assertEqual(self.owner.hold()[6],1.)

    def test_inflight_old_vla_and_tool_results_are_rejected_after_refine(self):
        self.owner.apply(self.command('refine'),24)
        self.assertFalse(self.owner.accepts(0,'vla',25))
        self.assertFalse(self.owner.accepts(0,'tool',25))
        self.assertTrue(self.owner.accepts(1,'vla',25))
        self.assertEqual(self.owner.stale_results,2)

    def test_latest_observation_replaces_old_one(self):
        for i in range(100): self.bus.publish({'step':i})
        self.assertEqual(self.bus.read()[0]['step'],99)

    def test_continue_with_new_vla_instruction_applies_and_invalidates_old_work(self):
        self.owner.actions.extend([np.ones(12) for _ in range(5)])
        self.owner.vla_pending=True
        guide=self.owner.apply(self.command('continue'),42)
        self.assertEqual(guide['instruction'],'retry grasping bread')
        self.assertEqual(guide['revision'],1)
        self.assertFalse(self.owner.actions)
        self.assertFalse(self.owner.accepts(0,'vla',43))
        self.assertEqual(self.owner.last_guidance_result['effective_command'],'refine')
        snap={key:np.zeros((16,16,3),np.uint8) for key in ['left','right','wrist']}
        snap['state']=np.zeros(16)
        self.assertEqual(policy_wire(snap,guide['instruction'])['prompt'],'retry grasping bread')

    def test_unchanged_continue_keeps_buffer_and_inflight_work(self):
        self.owner.actions.append(np.ones(12));self.owner.vla_pending=True
        for instruction in [' pick   bread ','']:
            self.owner.apply(self.command('continue',instruction=instruction,
                reason='Bread is now held; task is progressing.',memory='Continue the current goal.'),42)
            self.assertEqual(self.bus.guide['instruction'],'pick bread')
            self.assertEqual(self.bus.guide['revision'],0)
            self.assertEqual(len(self.owner.actions),1)
            self.assertTrue(self.owner.vla_pending)

    def test_continue_tool_change_is_applied_as_switch(self):
        guide=self.owner.apply(self.command('continue',tool='move_ee',args={'pose_ref':'current','offset_world':[0,0,.05]}),42)
        self.assertEqual(guide['tool'],'move_ee')
        self.assertEqual(self.owner.phase,'preparing')
        self.assertEqual(self.owner.last_guidance_result['effective_command'],'switch_tool')

    def test_continue_new_pose_restarts_but_new_description_does_not(self):
        args={'pose_ref':'current','offset_world':[0,0,.05]}
        self.owner.apply(self.command('switch_tool',tool='move_ee',args=args),40)
        motion=object();self.owner.motion=motion;self.owner.phase='move_ee'
        self.owner.apply(self.command('continue',tool='move_ee',revision=1,
            instruction='Move toward the requested pose',args=args),41)
        self.assertIs(self.owner.motion,motion)
        self.assertEqual(self.bus.guide['revision'],1)
        guide=self.owner.apply(self.command('continue',tool='move_ee',revision=1,
            args={'pose_ref':'current','offset_world':[0,0,.08]}),42)
        self.assertEqual(guide['args']['offset_world'],[0,0,.08])
        self.assertEqual(guide['revision'],2)
        self.assertIsNone(self.owner.motion)

    def test_unchanged_continue_cannot_resume_failed_or_finished_tool(self):
        self.owner.apply(self.command('switch_tool',tool='move_ee',args={'pose_ref':'current','offset_world':[0,0,.05]}),40)
        for phase in ['failed_move_ee','motion_done_unverified']:
            self.owner.phase=phase
            self.owner.apply(self.command('continue',tool='move_ee',revision=1,args={'pose_ref':'current','offset_world':[0,0,.05]}),41)
            self.assertEqual(self.owner.last_guidance_result['status'],'rejected')
            self.assertEqual(self.bus.guide['revision'],1)
            self.assertEqual(self.owner.phase,phase)

    def test_continue_cannot_bypass_required_tool_arguments(self):
        self.owner.apply(self.command('continue',tool='move_ee'),42)
        self.assertEqual(self.owner.last_guidance_result['status'],'rejected')
        self.assertEqual(self.bus.guide['tool'],'vla')

    def test_visual_assessments_do_not_override_vlm_continue_refine_or_hold(self):
        assessments=product(['normal','uncertain','stalled','failed'],
            ['unverified','confirmed','failed','lost'],['acquire','transport','place'],
            [('continue','vla'),('refine','vla'),('hold','hold')])
        for progress,grasp_status,stage,(command,tool) in assessments:
            with self.subTest(progress=progress,grasp_status=grasp_status,stage=stage,command=command):
                previous_tool=self.bus.guide['tool']
                revision=self.bus.guide['revision']
                instruction=self.bus.guide['instruction'] if command=='continue' else 'carry bread to bowl'
                self.owner.apply(self.command(command,tool=tool,revision=revision,
                    instruction=instruction,progress=progress,grasp_status=grasp_status,
                    stage=stage,target=''),42)
                self.assertEqual(self.bus.guide['tool'],tool)
                self.assertEqual(self.owner.last_guidance_result['status'],'applied')
                effective=self.owner.last_guidance_result['effective_command']
                self.assertEqual(effective,'switch_tool' if command=='continue' and tool!=previous_tool else command)

    def test_motion_selection_clears_previous_actions(self):
        self.owner.actions.extend([np.ones(12) for _ in range(5)])
        guide=self.owner.apply(self.command('switch_tool',tool='move_ee',
            args={'pose_ref':'current','offset_world':[0,0,.05]},progress='failed',grasp_status='failed'),42)
        self.assertEqual(guide['tool'],'move_ee')
        self.assertFalse(self.owner.actions)
        self.assertEqual(self.owner.phase,'preparing')

    def test_gripper_to_move_has_no_holding_or_failure_gate(self):
        self.owner.apply(self.command('switch_tool',tool='gripper',args={'command':'close'}),40)
        self.owner.phase='motion_done_unverified'
        guide=self.owner.apply(self.command('switch_tool',tool='move_ee',revision=1,
            args={'position_world':[.2,.1,1.]},progress='normal',grasp_status='unverified'),42)
        self.assertEqual(guide['tool'],'move_ee')
        self.assertEqual(self.owner.last_guidance_result['status'],'applied')

    def test_perception_and_its_result_do_not_change_vla_actions_or_gripper(self):
        self.owner.actions.append(np.ones(12));self.owner.vla_pending=True
        self.owner.last_close=1.
        request=self.owner.apply(self.command('switch_tool',tool='graspgen',args={'target':'bowl'}),42)
        self.assertTrue(request['query'])
        self.assertEqual(self.owner.last_guidance_result['effective_command'],'call_tool')
        self.assertEqual(self.bus.guide['revision'],0)
        self.assertEqual(self.bus.guide['tool'],'vla')
        self.assertEqual(len(self.owner.actions),1)
        self.assertTrue(self.owner.vla_pending)
        value={'pose_ref':'grasp:1','position_world':[0,0,1.],'quaternion_xyzw':[0,0,0,1]}
        # Even if VLM revises VLA during a query, a tagged read-only result can
        # be reported. It cannot start motion or change the new instruction.
        self.owner.apply(self.command('refine'),43)
        self.owner.actions.append(np.ones(12))
        self.owner.query_result(request,{'step':42,'wall_s':2.1},value,None,45)
        self.assertEqual(self.bus.guide['revision'],1)
        self.assertEqual(self.bus.guide['tool'],'vla')
        self.assertEqual(len(self.owner.actions),1)
        self.assertIsNone(self.owner.motion)
        self.assertEqual(self.owner.last_close,1.)
        self.assertEqual(self.owner.grasp_pose,value)
        self.assertEqual(self.owner.latest_tool_result['observed_step'],42)

    def test_perception_failure_reports_error_without_cancelling_active_motion(self):
        self.owner.apply(self.command('switch_tool',tool='move_ee',args={'position_world':[0,0,1.]}),40)
        motion=object();self.owner.motion=motion;self.owner.phase='move_ee'
        request=self.owner.apply(self.command('call_tool',tool='sam3',revision=1,args={'target':'bowl'}),41)
        self.owner.query_result(request,{'step':41},None,'No usable segmentation',42)
        self.assertIs(self.owner.motion,motion)
        self.assertEqual(self.owner.phase,'move_ee')
        self.assertEqual(self.owner.latest_tool_result['status'],'failed')
        self.assertFalse(self.owner.tool_pending)

    def test_slow_query_worker_does_not_block_observations_or_vla(self):
        entered=threading.Event();release=threading.Event()
        self.bus.publish({'step':1})
        self.owner.actions.extend([np.ones(12) for _ in range(5)])
        request=self.owner.apply(self.command('call_tool',tool='sam3',args={'target':'bowl'}),1)
        jobs=queue.Queue();results=queue.Queue()
        def infer(*args):
            entered.set()
            if not release.wait(3):raise TimeoutError('query test timed out')
            return {'request_id':1,'observed_step':1}
        with mock.patch('run_duplex.perceive',side_effect=infer):
            thread=threading.Thread(target=tool_worker,args=(self.bus,jobs,results,self.log),daemon=True)
            thread.start()
            try:
                jobs.put((request,{'step':1},None))
                self.assertTrue(entered.wait(3))
                for step in range(2,6):
                    self.bus.publish({'step':step})
                    self.owner.actions.popleft()
                self.assertEqual(self.bus.read()[0]['step'],5)
                self.assertEqual(self.bus.guide['tool'],'vla')
                release.set()
                kind,returned,source,value,error=results.get(timeout=3)
                self.assertEqual(kind,'query')
                self.owner.query_result(returned,source,value,error,5)
                self.assertEqual(self.owner.latest_tool_result['observed_step'],1)
                self.assertEqual(len(self.owner.actions),1)
                self.assertEqual(self.bus.guide['revision'],0)
            finally:
                release.set();self.bus.stop.set();thread.join(3)
        self.assertFalse(thread.is_alive())

    def test_busy_query_is_rejected_without_cancelling_active_motion(self):
        self.owner.apply(self.command('call_tool',tool='sam3',args={'target':'bowl'}),41)
        self.owner.apply(self.command('call_tool',tool='graspgen',args={'target':'bowl'}),42)
        self.assertEqual(self.owner.last_guidance_result['status'],'rejected')
        self.assertEqual(self.bus.guide['tool'],'vla')
        self.assertEqual(self.bus.guide['revision'],0)

    def test_native_gripper_sign_and_rolling_buffer_are_explicit(self):
        self.owner.last_close=-.999212
        action=np.zeros(12);action[6]=.999
        self.owner.actions.append(action)
        feedback=executor_feedback(self.bus.guide,self.owner,[.039,-.039],[0,0,0],20)
        self.assertEqual(feedback['gripper']['commanded'],'open')
        self.assertAlmostEqual(feedback['gripper']['finger_separation_m'],.078)
        self.assertNotIn('measured',feedback['gripper'])
        self.assertNotIn('holding_object',feedback['gripper'])
        self.assertEqual(feedback['action_buffer']['gripper_preview'],['close'])
        self.assertTrue(feedback['action_buffer']['auto_refilled'])
        self.assertFalse(feedback['action_buffer']['indicates_task_progress'])
        self.assertNotIn('last_gripper_close_command',feedback)
        self.owner.last_close=.999
        feedback=executor_feedback(self.bus.guide,self.owner,[.0014,-.0013],[0,0,0],20)
        self.assertEqual(feedback['gripper']['commanded'],'close')
        self.assertAlmostEqual(feedback['gripper']['finger_separation_m'],.0027)
        self.assertNotIn('measured',feedback['gripper'])

    def test_slow_vlm_does_not_block_capture_and_next_request_uses_latest_frame(self):
        entered=threading.Event();release=threading.Event();second=threading.Event()
        requests=[];errors=[]
        def frame(step):
            return {'step':step,'wall_s':step/20,'captured_at_monotonic':time.monotonic(),
                'feedback':{'phase':'executing'},
                **{k:np.full((16,16,3),step,np.uint8) for k in ['left','right','wrist']}}
        def post(url,**kwargs):
            requests.append(kwargs['json'])
            if len(requests)==1:
                entered.set()
                if not release.wait(3): raise TimeoutError('Test did not release model response')
            else:
                second.set();self.bus.stop.set()
            decision={'command':'continue','tool':'vla','instruction':'pick bread','args':{},
                'plan':['grasp bread','verify holding','place bread']}
            response=mock.Mock()
            response.json.return_value={'choices':[{'message':{'content':json.dumps(decision)}}]}
            return response
        def supervise_checked():
            try: supervise(self.bus,'pick bread',self.log)
            except Exception as exc: errors.append(exc)
        self.bus.publish(frame(0))
        with mock.patch('duplex_vlm.httpx.Client') as client:
            client.return_value.__enter__.return_value.post.side_effect=post
            thread=threading.Thread(target=supervise_checked,daemon=True);thread.start()
            try:
                self.assertTrue(entered.wait(3))
                for step in range(1,21): self.bus.publish(frame(step))
                self.assertEqual(self.bus.read()[0]['step'],20)
                self.assertEqual(json.loads(requests[0]['messages'][1]['content'][0]['text'])['observed_step'],0)
                release.set()
                with self.bus.condition:
                    self.assertTrue(self.bus.condition.wait_for(lambda: bool(self.bus.commands),timeout=3))
                command=self.bus.take_commands()[0]
                self.owner.apply(command,20)
                latest=frame(21);latest['feedback']['last_guidance_result']=self.owner.last_guidance_result
                self.bus.publish(latest);self.bus.acknowledge(command['command_id'])
                self.assertTrue(second.wait(3))
            finally:
                release.set();self.bus.stop.set()
                with self.bus.condition:self.bus.condition.notify_all()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        next_input=json.loads(requests[1]['messages'][1]['content'][0]['text'])
        self.assertEqual(next_input['observed_step'],21)
        self.assertEqual(next_input['feedback']['last_guidance_result']['status'],'applied')
        for request in requests[:2]:
            self.assertEqual(request['model'],'gpt-6.1-sol')
            self.assertEqual(request['reasoning_effort'],'high')
            self.assertEqual(request['service_tier'],'fast')
            self.assertNotIn('thinking',request)
        self.assertEqual(requests[1]['max_tokens'],1024)
        self.assertEqual(next_input['request_kind'],'monitor')
        self.assertEqual(next_input['initial_plan'],['grasp bread','verify holding','place bread'])
        self.log.file.flush()
        events=[json.loads(line) for line in (Path(self.temp.name)/'events.jsonl').read_text().splitlines()]
        first_end=next(event for event in events if event['event']=='vlm_end')
        self.assertEqual(first_end['observed_step'],0)
        self.assertEqual(first_end['current_step'],20)
        self.assertGreaterEqual(first_end['input_frame_age_s'],first_end['latest_frame_age_s'])

    def test_dequeue_does_not_acknowledge_before_new_state_is_published(self):
        command_id=self.bus.command({'command':'hold','tool':'hold'},0,5)
        self.bus.take_commands()
        self.assertLess(self.bus.acknowledged,command_id)
        self.bus.publish({'step':6,'revision':1})
        self.bus.acknowledge(command_id)
        self.assertEqual(self.bus.acknowledged,command_id)

    def test_geometric_action_uses_native_rotated_base_frame(self):
        goal=Waypoint('approach',np.array([.02,0,0]),np.eye(3),1.)
        motion=MotionExecutor(ToolPlan('move_ee',[goal],{}))
        origin=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        action=motion.tick({'ee_pos':np.zeros(3),'ee_rot':np.eye(3),
            'controller_origin_rot':origin,'controller_scale':np.array([.05]*3+[.5]*3)})
        self.assertAlmostEqual(action[0],0)
        self.assertLess(action[1],0)
        self.assertEqual(action[6],1)

    def test_completed_motion_is_not_reported_as_verified_grasp(self):
        motion=MotionExecutor(ToolPlan('move_ee',[Waypoint('lift',np.zeros(3),np.eye(3),1.)],{}))
        action=motion.tick({'ee_pos':np.zeros(3),'ee_rot':np.eye(3)})
        self.assertIsNone(action)
        self.assertEqual(motion.status,'motion_done_unverified')

    def test_graspgen_returns_pose_without_a_robot_action_sequence(self):
        pose=np.eye(4);pose[:3,:3]=np.diag([1,-1,-1]);pose[2,3]=.9
        client=mock.Mock()
        client.detect_and_segment.return_value=(np.ones((16,16),bool),.9,(0,0,16,16))
        client.compute_grasp.return_value=(pose,.8)
        mount=np.eye(4);mount[2,3]=.097
        snap={'step':0,'left':np.zeros((16,16,3),np.uint8),'depth':np.full((16,16),.8),
            'K':np.array([[16,0,8],[0,16,8],[0,0,1]]),'T_camera':np.eye(4),
            'ee_pos':np.array([-.3,0,1.3]),'ee_rot':np.diag([1,-1,-1]),'T_hand_to_site':mount,
            'gripper_command':1.}
        with mock.patch.dict('os.environ',{'RUN_DIR':self.temp.name}):
            result=perceive({'tool':'graspgen','revision':0,'request_id':1,'args':{'target':'bowl'}},snap,self.log,client=client)
        self.assertAlmostEqual(result['position_world'][2],.803)
        self.assertEqual(result['pose_ref'],'grasp:1')
        self.assertNotIn('waypoints',result)
        self.assertFalse(client.compute_grasp.call_args.kwargs['enable_topdown_filter'])
        plan=make_plan({'tool':'move_ee','revision':1,'args':{'pose_ref':'grasp:1'}},snap,self.log,result)
        self.assertEqual(len(plan.waypoints),1)
        goal=plan.waypoints[0]
        self.assertEqual(goal.phase,'move_ee')
        self.assertAlmostEqual(goal.pos[2],.803)
        self.assertEqual(goal.close,1.)
        self.assertIsNone(goal.timeout_s)

    def test_pose_offset_and_gripper_change_only_when_requested(self):
        snap={'ee_pos':np.array([1.,2.,3.]),'ee_rot':np.eye(3),'gripper_command':1.}
        plan=make_plan({'tool':'move_ee','revision':1,'args':{
            'pose_ref':'current','offset_world':[0,0,.03]}},snap,self.log)
        np.testing.assert_allclose(plan.waypoints[0].pos,[1,2,3.03])
        self.assertEqual(plan.waypoints[0].close,1.)
        plan=make_plan({'tool':'move_ee','revision':1,'args':{
            'position_world':[1,2,3],'gripper':'open','timeout_s':2.}},snap,self.log)
        self.assertEqual(plan.waypoints[0].close,0.)
        self.assertEqual(plan.waypoints[0].timeout_s,2.)
        with self.assertRaises(ValueError):
            make_plan({'tool':'move_ee','revision':1,'args':{'pose_ref':'grasp:99'}},snap,self.log)

    def test_gripper_does_not_add_lift_or_transport(self):
        snap={'ee_pos':np.zeros(3),'ee_rot':np.eye(3),'gripper_command':0.}
        plan=make_plan({'tool':'gripper','revision':1,'args':{'command':'close','duration_s':.1}},snap,self.log)
        motion=MotionExecutor(plan)
        with mock.patch('robocasa_tools.time.monotonic',return_value=motion.started+.05):
            action=motion.tick(snap)
        self.assertEqual(action[6],1.)
        np.testing.assert_array_equal(np.delete(action,6),np.zeros(11))
        with mock.patch('robocasa_tools.time.monotonic',return_value=motion.started+.2):
            self.assertIsNone(motion.tick(snap))
        self.assertEqual(motion.status,'motion_done_unverified')

    def test_motion_at_goal_still_applies_explicit_gripper_change(self):
        snap={'ee_pos':np.zeros(3),'ee_rot':np.eye(3),'gripper_command':0.}
        plan=make_plan({'tool':'move_ee','revision':1,'args':{
            'pose_ref':'current','gripper':'close'}},snap,self.log)
        action=MotionExecutor(plan).tick(snap)
        self.assertEqual(action[6],1.)
        np.testing.assert_array_equal(np.delete(action,6),np.zeros(11))

    def test_video_keeps_wall_clock_pause(self):
        import cv2
        path=Path(self.temp.name)/'video.mp4'
        video=TimedVideo(path,self.log,fps=20)
        video.update(np.zeros((64,64,3),dtype=np.uint8),{'phase':'waiting_for_vlm'})
        video.start(time.monotonic())
        time.sleep(.55)
        report=video.close(time.monotonic())
        cap=cv2.VideoCapture(str(path))
        duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS);cap.release()
        self.assertLess(abs(duration-report['wall_from_first_action_s']),.08)
        self.assertIsNone(report['error'])

    def test_missing_tool_preserves_unambiguous_command_meaning(self):
        value=parse_decision('{"command":"continue","instruction":""}','move_ee')
        self.assertEqual(value['tool'],'move_ee')
        self.assertTrue(value['tool_defaulted'])
        value=parse_decision('{"command":"refine","instruction":"Place the broccoli in the bowl."}','hold')
        self.assertEqual(value['tool'],'vla')
        for command in ['call_tool','switch_tool']:
            with self.assertRaises(ValueError):
                parse_decision(json.dumps({'command':command,'args':{'target':'bowl'}}),'vla')

    def test_response_argument_error_is_visible_to_the_next_model_call(self):
        requests=[]
        def frame(step):
            return {'step':step,'wall_s':step,'captured_at_monotonic':time.monotonic(),
                'feedback':{'phase':'executing'},
                **{k:np.zeros((16,16,3),np.uint8) for k in ['left','right','wrist']}}
        def post(url,**kwargs):
            requests.append(kwargs['json'])
            response=mock.Mock()
            if len(requests)==1:
                decision={'command':'switch_tool','tool':'sam3','args':{}}
                self.bus.publish(frame(1))
            else:
                decision={'command':'continue','tool':'vla','instruction':''}
                self.bus.stop.set()
            response.json.return_value={'choices':[{'message':{'content':json.dumps(decision)}}]}
            return response
        self.bus.publish(frame(0))
        with mock.patch('duplex_vlm.httpx.Client') as client:
            client.return_value.__enter__.return_value.post.side_effect=post
            supervise(self.bus,'pick bread',self.log)
        self.assertEqual(len(requests),2)
        state=json.loads(requests[1]['messages'][1]['content'][0]['text'])
        self.assertIn('Missing tool argument: target',state['last_response_error'])
        self.assertEqual(self.bus.guide['tool'],'vla')

    def test_historical_visual_baseline_is_retained_across_goal_changes(self):
        history=[{'step':0,'wall_s':0,'instruction':'pick bread'},
            {'step':200,'wall_s':10,'instruction':'pick bread'},
            {'step':360,'wall_s':18,'instruction':'place bread'}]
        reference=progress_reference(history,21)
        self.assertEqual(reference['step'],200)
        self.assertEqual(reference['instruction'],'pick bread')
        self.assertIsNone(progress_reference(history[:1],5))

    def test_invalid_tool_decision_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_decision('{"command":"call_tool","tool":"graspgen","args":{}}')


if __name__=='__main__': unittest.main()
