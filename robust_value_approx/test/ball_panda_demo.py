import pybullet as p
import time
import os
import inspect
import torch
import numpy as np
import robust_value_approx.ball_paddle_hybrid_linear_system as bp
import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.constants as constants
import matplotlib.pyplot as plt
import math


if __name__ == "__main__":
    dtype = torch.float64
    dt = .1
    ball_capture = torch.Tensor([.5, .5]).type(dtype)
    capture_size = .05
    paddle_ang = torch.Tensor([.1, .2, .3, .4, .5, .6, .7]).type(dtype)
    x_lo = torch.Tensor([-10., -10., 0., -np.pi, -1e2, -1e2, 0.]).type(dtype)
    x_up = torch.Tensor([10., 10., 1., np.pi, 1e2, 1e2, 1e2]).type(dtype)
    u_lo = torch.Tensor([-1e2, -1e2]).type(dtype)
    u_up = torch.Tensor([1e2, 1e2]).type(dtype)
    b_cap_lo = ball_capture - capture_size
    b_cap_up = ball_capture + capture_size
    sys_con = bp.get_ball_paddle_hybrid_linear_system_vel_ctrl
    sys = sys_con(dtype, dt,
                  x_lo, x_up,
                  u_lo, u_up,
                  ball_capture_lo=b_cap_lo,
                  ball_capture_up=b_cap_up,
                  paddle_angles=paddle_ang,
                  cr=.9, collision_eps=.01,
                  midpoint=True)
    N = 10
    vf = value_to_optimization.ValueFunction(sys, N, x_lo, x_up, u_lo, u_up)
    Q = torch.diag(torch.Tensor([1., 1., 0., 0., 0., 0., 0.]).type(dtype))
    Qt = torch.diag(torch.Tensor([100., 100., 0., 0., 0., 0., 0.]).type(dtype))
    R = torch.diag(torch.Tensor([.1, .001]).type(dtype))
    Rt = torch.diag(torch.Tensor([.1, .001]).type(dtype))
    xtraj = torch.Tensor([ball_capture[0], ball_capture[1], 0., 0.,
                          0., 0., 0.]).type(dtype).unsqueeze(1).repeat(1, N-1)
    vf.set_cost(Q=Q, R=R)
    vf.set_terminal_cost(Qt=Qt, Rt=Rt)
    vf.set_traj(xtraj=xtraj)
    V = vf.get_value_function()

    # # generate an initial state
    bally0 = .75
    paddley0 = .15
    paddletheta0 = 0.
    ballvy0 = -2.  # anywhere between -5 and 0 seems OK
    x0 = torch.Tensor([0., bally0, paddley0, paddletheta0,
                       0., ballvy0, 0.]).type(dtype)
    # compute a feedforward trajectory from vf
    (obj_val, s_val, alpha_val) = V(x0)
    traj_val = torch.cat((x0, s_val)).reshape(N, -1).t()
    xtraj_val = traj_val[:sys.x_dim, :]
    utraj_val = traj_val[sys.x_dim:sys.x_dim+sys.u_dim, :]
    ttraj_val = torch.arange(0., N*dt, dt, dtype=dtype)

    p.connect(p.GUI)
    p.setGravity(0, 0, constants.G)
    currentdir = os.path.dirname(os.path.abspath(
                                 inspect.getfile(inspect.currentframe())))
    p.setAdditionalSearchPath(currentdir)
    plane_id = p.loadURDF("ball_panda_description/plane.urdf", [0, 0, 0])
    panda_id = p.loadURDF("ball_panda_description/panda.urdf", [-.5, 0, 0])
    panda_num_joints = 7
    # the true joint limits
    # panda_ll = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -.0873, -2.9671]
    # panda_ul = [2.9671, 1.8326, 2.9671, 0., 2.9671, 3.8223, 2.9671]
    # joint limits to get nice IK solutions
    panda_ll = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -.0873, -2.9671]
    panda_ul = [2.9671, -.5, 2.9671, 0., 2.9671, 3.8223, 2.9671]
    panda_jr = [2.*math.pi] * panda_num_joints
    panda_rp = [0., -.785, 0., -2.356, 0., 1.571, .785]
    # panda_efforts = [87., 87., 87., 87., 12., 12., 12.]
    panda_efforts = [8700., 8700., 8700., 8700., 1200., 1200., 1200.]
    pos = [0., 0., paddley0]
    orn = p.getQuaternionFromEuler([.5*math.pi + paddletheta0, 0., .5*math.pi])
    joint_poses = p.calculateInverseKinematics(panda_id,
                                               12,
                                               pos,
                                               orn,
                                               lowerLimits=panda_ll,
                                               upperLimits=panda_ul,
                                               jointRanges=panda_jr,
                                               restPoses=panda_rp,
                                               solver=p.IK_DLS)
    for i in range(panda_num_joints):
        p.resetJointState(panda_id, i, joint_poses[i], 0.)
    ball_id = p.loadURDF("ball_panda_description/ball.urdf", [0, 0, bally0])
    p.resetBaseVelocity(ball_id, [0, 0, ballvy0])
    target_vis_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                        rgbaColor=[0, 1, 1, .75],
                                        specularColor=[.4, .4, 0],
                                        radius=capture_size)
    target_col_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                           radius=capture_size)
    p.createMultiBody(baseVisualShapeIndex=target_vis_id,
                      baseCollisionShapeIndex=target_col_id,
                      basePosition=[ball_capture[0], 0., ball_capture[1]])
    p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.)
    p.stepSimulation()
    dt_sim = .001
    p.setTimeStep(dt_sim)
    ttraj_val_sim = torch.arange(0., N*dt, dt_sim, dtype=dtype)
    ttraj_val_np = ttraj_val.detach().numpy()
    xtraj_val_np = xtraj_val.detach().numpy()
    utraj_val_np = utraj_val.detach().numpy()
    ttraj_val_sim_np = ttraj_val_sim.detach().numpy()
    xtraj_val_sim = torch.zeros(xtraj_val.shape[0], ttraj_val_sim.shape[0])
    utraj_val_sim = torch.zeros(utraj_val.shape[0], ttraj_val_sim.shape[0])
    for i in range(xtraj_val.shape[0]):
        xtraj_val_sim[i, :] = torch.Tensor(
            np.interp(ttraj_val_sim_np,
                      ttraj_val_np, xtraj_val_np[i, :])).type(dtype)
    for i in range(utraj_val.shape[0]):
        utraj_val_sim[i, :] = torch.Tensor(
            np.interp(ttraj_val_sim_np,
                      ttraj_val_np, utraj_val_np[i, :])).type(dtype)
    xtraj_sim = torch.zeros((sys.x_dim, ttraj_val_sim.shape[0]), dtype=dtype)
    for n in range(ttraj_val_sim.shape[0]):
        ball_pos = p.getBasePositionAndOrientation(ball_id)[0]
        (ballx, bally) = (ball_pos[0], ball_pos[2])
        ball_vel = p.getBaseVelocity(ball_id)[0]
        (ballvx, ballvy) = (ball_vel[0], ball_vel[2])
        paddle_state = p.getLinkState(panda_id, 12,
                                      computeLinkVelocity=1,
                                      computeForwardKinematics=1)
        (paddley, paddlevy) = (paddle_state[0][2], paddle_state[6][2])
        paddletheta = p.getEulerFromQuaternion(paddle_state[1])[0] - .5*math.pi
        x_current = torch.Tensor([ballx, bally, paddley, paddletheta,
                                  ballvx, ballvy, paddlevy]).type(dtype)
        xtraj_sim[:, n] = x_current
        pos = [0., 0., xtraj_val_sim[2, n]]
        orn = p.getQuaternionFromEuler([.5*math.pi + xtraj_val_sim[3, n],
                                        0.,
                                        .5*math.pi])
        joint_poses = p.calculateInverseKinematics(panda_id,
                                                   12,
                                                   pos,
                                                   orn,
                                                   lowerLimits=panda_ll,
                                                   upperLimits=panda_ul,
                                                   jointRanges=panda_jr,
                                                   restPoses=joint_poses,
                                                   solver=p.IK_DLS)
        lin_jac, ang_jac = p.calculateJacobian(panda_id,
                                               12,
                                               [0., 0., 0.],
                                               joint_poses,
                                               [0.] * panda_num_joints,
                                               [0.] * panda_num_joints)
        lin_jac = torch.Tensor(lin_jac).type(dtype)
        ee_vels = torch.Tensor([0., 0., xtraj_val_sim[6, n]]).type(dtype)
        joint_vels = torch.pinverse(lin_jac) @ ee_vels
        p.setJointMotorControlArray(panda_id,
                                    jointIndices=range(panda_num_joints),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_poses,
                                    targetVelocities=joint_vels,
                                    forces=panda_efforts,
                                    positionGains=1.*torch.ones(
                                        panda_num_joints),
                                    velocityGains=1.*torch.ones(
                                        panda_num_joints))
        p.stepSimulation()
        time.sleep(0.001)
    legend = []
    plt.plot(ttraj_val, xtraj_val[0, :], '*')
    legend.append('plan ballx')
    plt.plot(ttraj_val, xtraj_val[1, :], '*')
    legend.append('plan bally')
    plt.plot(ttraj_val, xtraj_val[2, :], '*')
    legend.append('plan paddle y')
    plt.plot(ttraj_val, xtraj_val[3, :], '*')
    legend.append('plan paddle theta')
    # plt.plot(ttraj_val, xtraj_val[6,:],'*')
    # legend.append('plan paddlev')
    plt.plot(ttraj_val_sim, xtraj_sim[0, :])
    legend.append('ballx')
    plt.plot(ttraj_val_sim, xtraj_sim[1, :])
    legend.append('bally')
    plt.plot(ttraj_val_sim, xtraj_sim[2, :])
    legend.append('paddle y')
    plt.plot(ttraj_val_sim, xtraj_sim[3, :])
    legend.append('paddle theta')
    # plt.plot(ttraj_val_sim, xtraj_sim[6,:])
    # legend.append('paddlev')
    plt.legend(legend)
    plt.show()
