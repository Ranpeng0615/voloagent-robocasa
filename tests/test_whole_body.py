"""Whole-body allocation and native action-frame invariants."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from whole_body import whole_body_action
from robocasa_tools import MotionExecutor,ToolPlan,Waypoint


class WholeBodyTests(unittest.TestCase):
    def snapshot(self):
        jac=np.zeros((6,11));jac[0,0]=jac[1,1]=jac[5,2]=jac[2,3]=1.
        return {'ee_pos':np.zeros(3),'ee_rot':np.eye(3),
            'controller_origin_rot':np.eye(3),'controller_scale':np.array([.05]*3+[.5]*3),
            'kinematics':{'jacobian':jac,'q':np.zeros(11),'limits':np.tile([-np.inf,np.inf],(11,1)),
                'dt':.05,'base_action_to_qvel':np.eye(3),'torso_scale':.05}}

    def test_base_can_reach_direction_unavailable_to_arm(self):
        snap=self.snapshot()
        action,diagnostics=whole_body_action(snap,np.array([.1,0,0]),np.zeros(3),0,np.zeros(3))
        self.assertGreater(action[7],0)
        np.testing.assert_array_equal(action[:6],np.zeros(6))
        self.assertEqual(diagnostics['ik_mode'],'whole_body_differential')
        self.assertLess(diagnostics['predicted_position_residual_m'],.010)

    def test_base_velocity_uses_inverse_native_controller_frame(self):
        snap=self.snapshot();snap['kinematics']['base_action_to_qvel']=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        action,_=whole_body_action(snap,np.array([.1,0,0]),np.zeros(3),0,np.zeros(3))
        self.assertAlmostEqual(action[7],0)
        self.assertLess(action[8],0)

    def test_base_is_stationary_during_finger_closure(self):
        snap=self.snapshot()
        action,_=whole_body_action(snap,np.array([.1,0,0]),np.zeros(3),1,np.zeros(3),base_enabled=False)
        np.testing.assert_array_equal(action[7:10],np.zeros(3))
        self.assertEqual(action[6],1)

    def test_native_friction_is_compensated_but_stop_stays_zero(self):
        snap=self.snapshot();snap['kinematics']['base_friction_velocity']=np.array([.25,.25,1/6])
        action,_=whole_body_action(snap,np.array([.1,0,0]),np.zeros(3),0,np.zeros(3))
        self.assertGreater(action[7],.25)
        stopped,_=whole_body_action(snap,np.zeros(3),np.zeros(3),0,np.zeros(3))
        np.testing.assert_array_equal(stopped[7:10],np.zeros(3))

    def test_torso_joint_limit_is_respected(self):
        snap=self.snapshot();snap['kinematics']['limits'][3]=[0,0]
        action,_=whole_body_action(snap,np.array([0,0,.1]),np.zeros(3),0,np.zeros(3))
        self.assertEqual(action[10],0)

    def test_native_base_joint_limit_prevents_outward_motion(self):
        snap=self.snapshot();snap['kinematics']['q'][0]=.35
        snap['kinematics']['limits'][0]=[0,.35]
        action,_=whole_body_action(snap,np.array([.1,0,0]),np.zeros(3),0,np.zeros(3))
        self.assertLessEqual(action[7],1e-8)

    def test_no_adapter_displacement_budget_blocks_base_motion(self):
        snap=self.snapshot();snap['kinematics']['q'][0]=.4
        action,_=whole_body_action(snap,np.array([.1,0,0]),np.zeros(3),0,np.zeros(3))
        self.assertGreater(action[7],0)

    def test_tool_executor_uses_whole_body_solver(self):
        motion=MotionExecutor(ToolPlan('move_ee',[Waypoint('move_ee',np.array([.1,0,0]),np.eye(3),0)],{}))
        action=motion.tick(self.snapshot())
        self.assertGreater(action[7],0)
        self.assertEqual(motion.tracking_error['ik_mode'],'whole_body_differential')


if __name__=='__main__':unittest.main()
