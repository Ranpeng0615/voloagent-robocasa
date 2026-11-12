"""Local differential IK for the native PandaOmron base, torso and arm.

Robot-only Jacobians are copied on the control thread. No worker reads MuJoCo.
This allocates one Cartesian tracking step; it is not global navigation or a
collision-free trajectory planner. The native VLA action interface is retained.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear
from robosuite.utils import transform_utils as T


class RobotKinematics:
    def __init__(self, robot, control_hz):
        self.robot=robot;self.dt=1/float(control_hz)
        parts=robot.composite_controller.part_controllers
        self.base,self.torso,self.arm=(parts[k] for k in ['base','torso','right'])
        controllers=[self.base,self.torso,self.arm]
        self.joints=np.concatenate([c.joint_index for c in controllers]).astype(int)
        self.qpos=np.concatenate([c.qpos_index for c in controllers]).astype(int)
        self.dofs=np.concatenate([c.qvel_index for c in controllers]).astype(int)
        if [len(c.qvel_index) for c in controllers]!=[3,1,7]:
            raise ValueError('Expected PandaOmron base(3), torso(1), arm(7)')
        if self.torso.input_type!='delta':raise ValueError('Expected native delta torso control')
        self.base_type=type(self.base).__name__
        if self.base_type not in {'LegacyMobileBaseJointVelocityController','MobileBaseJointVelocityController'}:
            raise ValueError('Unsupported native mobile base controller')
        self.reversed_axis=(self.base._check_forward_joint_reversed()
            if self.base_type.startswith('Legacy') else False)
        self.weight=(self.base.actuator_max-self.base.actuator_min)/2
        if not np.allclose(self.base.actuator_max+self.base.actuator_min,0):
            raise ValueError('Expected symmetric base velocity actuators')
        self.site=robot.sim.model.site_id2name(robot.eef_site_id['right'])
        model=robot.sim.model
        actuators=[int(np.flatnonzero(model.actuator_trnid[:,0]==j)[0]) for j in self.joints[:3]]
        self.base_friction_velocity=model.dof_frictionloss[self.dofs[:3]]/model.actuator_gainprm[actuators,0]

    def capture(self):
        sim=self.robot.sim;model,data=sim.model,sim.data
        jac=np.vstack([data.get_site_jacp(self.site),data.get_site_jacr(self.site)])[:,self.dofs]
        # Match the pinned native controller's initial/current base frame and
        # its legacy axis convention, including actuator velocity scaling.
        _,orientation=self.base.get_base_pose()
        theta=T.mat2euler(orientation)[2]-T.mat2euler(self.base.init_ori)[2]
        if self.base_type.startswith('Legacy') and not self.reversed_axis:
            angle=theta
        else:angle=-theta
        co,si=np.cos(angle),np.sin(angle)
        rotation=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
        if self.reversed_axis:rotation=rotation[:,[1,0,2]]
        limited=model.jnt_limited[self.joints].astype(bool)
        limits=model.jnt_range[self.joints].copy()
        limits[~limited]=[-np.inf,np.inf]
        return {'jacobian':jac.copy(),'q':data.qpos[self.qpos].copy(),
            'limits':limits,'dt':self.dt,
            'base_action_to_qvel':np.diag(self.weight)@rotation,
            'base_friction_velocity':self.base_friction_velocity.copy(),
            'torso_scale':float((self.torso.output_max-self.torso.output_min)[0]/
                (self.torso.input_max-self.torso.input_min)[0])}


def limit_norm(value, maximum):
    return value*min(1.,maximum/max(float(np.linalg.norm(value)),1e-12))


def whole_body_action(snapshot, dp, dr, close, anchor, base_enabled=True):
    """Bounded, damped least squares over all 11 robot configuration DOFs.

    Arm increments become OSC pose deltas; base increments become native
    velocity actions; torso increments become native joint-position deltas.
    """
    k=snapshot['kinematics'];jac=k['jacobian'];q=k['q'];dt=k['dt']
    caps=np.array([.12,.12,.25,.03]+[1.]*7)*dt
    lower=np.maximum(-caps,np.minimum(caps,k['limits'][:,0]-q))
    upper=np.minimum(caps,np.maximum(-caps,k['limits'][:,1]-q))
    # Local repositioning around the tool's starting base, not room navigation.
    budget=np.array([.35,.35,.5])
    lower[:3]=np.maximum(lower[:3],np.clip(anchor-budget-q[:3],-caps[:3],caps[:3]))
    upper[:3]=np.minimum(upper[:3],np.clip(anchor+budget-q[:3],-caps[:3],caps[:3]))
    if not base_enabled:lower[:3]=upper[:3]=0.
    desired=np.r_[limit_norm(dp,.010),limit_norm(dr,.080)]
    task_weight=np.array([1.,1.,1.,.2,.2,.2])
    normalized=jac*caps[None,:]
    regularizer=.001*np.diag([2.,2.,2.,1.2]+[1.]*7)
    matrix=np.vstack([task_weight[:,None]*normalized,regularizer])
    rhs=np.r_[task_weight*desired,np.zeros(11)]
    lo,hi=lower/caps,upper/caps
    active=hi-lo>1e-10
    z=(lo+hi)/2
    if active.any():
        solution=lsq_linear(matrix[:,active],rhs-matrix[:,~active]@z[~active],
            bounds=(lo[active],hi[active]),tol=1e-6,max_iter=40)
        z[active]=solution.x
    dq=z*caps
    arm_delta=jac[:,4:]@dq[4:]
    origin=snapshot['controller_origin_rot']
    action=np.zeros(12)
    action[:6]=np.clip(np.r_[origin.T@arm_delta[:3],origin.T@arm_delta[3:]]/
        snapshot['controller_scale'],-1,1)
    action[6]=close
    base_velocity=dq[:3]/dt
    # Native velocity servos otherwise lose small commands to Coulomb friction
    # (250 / 1000 = .25 m/s for PandaOmron translation). Feed forward only
    # while motion is requested; zero remains a true stop command.
    friction=np.asarray(k.get('base_friction_velocity',np.zeros(3)))
    compensation=np.where(np.abs(base_velocity)>1e-4,np.sign(base_velocity)*friction,0.)
    action[7:10]=np.clip(np.linalg.solve(k['base_action_to_qvel'],base_velocity+compensation),-1,1)
    action[10]=np.clip(dq[3]/k['torso_scale'],-1,1)
    # Achieved-pose mode: remeasure each step, avoiding accumulated OSC goals.
    action[11]=0.
    diagnostics={'ik_mode':'whole_body_differential','base_action':action[7:10].tolist(),
        'torso_action':float(action[10]),'base_displacement_m':float(np.linalg.norm(q[:2]-anchor[:2])),
        'predicted_position_residual_m':float(np.linalg.norm((jac@dq-desired)[:3]))}
    return action,diagnostics
