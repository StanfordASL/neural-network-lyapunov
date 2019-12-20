import pybullet as p
import time
import os
import inspect
import torch
import numpy as np
import robust_value_approx.ball_paddle_hybrid_linear_system as bphls
import robust_value_approx.value_to_optimization as value_to_optimization
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dtype = torch.float64
    dt = .1
    ball_capture = torch.Tensor([.5, .5]).type(dtype)
    capture_size = .05
    paddle_ang = torch.Tensor([.1, .2, .3, .4, .5, .6, .7]).type(dtype)
    x_lo = torch.Tensor([-10., -10., 0., -1e2, -1e2, 0.]).type(dtype)
    x_up = torch.Tensor([10., 10., 1., 1e2, 1e2, 1e2]).type(dtype)
    u_lo = torch.Tensor([-np.pi, -1e4]).type(dtype)
    u_up = torch.Tensor([np.pi, 1e4]).type(dtype)
    b_cap_lo = ball_capture - capture_size
    b_cap_up = ball_capture + capture_size
    sys = bphls.get_ball_paddle_hybrid_linear_system(dtype, dt, x_lo, x_up,
                                                     u_lo, u_up,
                                                     ball_capture_lo=b_cap_lo,
                                                     ball_capture_up=b_cap_up,
                                                     paddle_angles=paddle_ang,
                                                     cr=.9, collision_eps=.01,
                                                     midpoint=True)
    N = 10
    vf = value_to_optimization.ValueFunction(sys, N, x_lo, x_up, u_lo, u_up)
    Q = torch.diag(torch.Tensor([1., 1., 0., 0., 0., 0.]).type(dtype))
    Qt = torch.diag(torch.Tensor([100., 100., 0., 0., 0., 0.]).type(dtype))
    R = torch.diag(torch.Tensor([1., .001]).type(dtype))
    Rt = torch.diag(torch.Tensor([1., .001]).type(dtype))
    xtraj = torch.Tensor([ball_capture[0], ball_capture[1], 0.,
                          0., 0., 0.]).type(dtype).unsqueeze(1).repeat(1, N-1)
    vf.set_cost(Q=Q, R=R)
    vf.set_terminal_cost(Qt=Qt, Rt=Rt)
    vf.set_traj(xtraj=xtraj)
    # to make it easier we set the paddle angle to be constant
    vf.set_constant_control(0)
    V = vf.get_value_function()

    # generate an initial state
    bally0 = .75
    paddley0 = .15
    ballvy0 = -2. # anywhere between -5 and 0 seems OK
    x0 = torch.Tensor([0., bally0, paddley0, 0., ballvy0, 0.]).type(dtype)
    # compute a feedforward trajectory from vf
    (obj_val, s_val, alpha_val) = V(x0)
    traj_val = torch.cat((x0, s_val)).reshape(N, -1).t()
    xtraj_val = traj_val[:sys.x_dim, :]
    utraj_val = traj_val[sys.x_dim:sys.x_dim+sys.u_dim, :]
    ttraj_val = torch.arange(0., N*dt, dt, dtype=dtype)

    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    currentdir = os.path.dirname(os.path.abspath(
                                 inspect.getfile(inspect.currentframe())))
    p.setAdditionalSearchPath(currentdir)
    plane_id = p.loadURDF("ball_paddle_description/plane.urdf", [0., 0., 0.])
    paddle_id = p.loadURDF(
        "ball_paddle_description/paddle.urdf", [-.125, 0, 0])
    p.resetJointState(paddle_id, 0, paddley0, targetVelocity=0.)
    p.resetJointState(paddle_id, 1, utraj_val[0, 0], targetVelocity=0.)
    ball_id = p.loadURDF("ball_paddle_description/ball.urdf", [0, 0, bally0])
    p.resetBaseVelocity(ball_id, [0, 0, ballvy0])
    target_vis_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                        rgbaColor=[0, 1, 1, .75],
                                        specularColor=[.4, .4, 0],
                                        radius=capture_size)
    target_col_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                           radius=capture_size)
    p.createMultiBody(baseVisualShapeIndex=target_vis_id,
                      baseCollisionShapeIndex=target_col_id,
                      basePosition=[0., ball_capture[0], ball_capture[1]])
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
        (ballx, bally) = p.getBasePositionAndOrientation(ball_id)[0][1:3]
        (ballvx, ballvy) = p.getBaseVelocity(ball_id)[0][1:3]
        (paddley, paddlevy) = p.getJointState(paddle_id, 0)[0:2]
        x_current = torch.Tensor(
            [ballx, bally, paddley, ballvx, ballvy, paddlevy]).type(dtype)
        xtraj_sim[:, n] = x_current
        p.setJointMotorControlArray(paddle_id,
                                    jointIndices=[0, 1],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[
                                        xtraj_val_sim[2, n],
                                        utraj_val_sim[0, n]],
                                    targetVelocities=[xtraj_val_sim[5, n], 0.],
                                    forces=[1e9, 1e9],
                                    positionGains=[1., 1.],
                                    velocityGains=[1., 1.])
        p.stepSimulation()
        time.sleep(0.001)
    legend = []
    plt.plot(ttraj_val, xtraj_val[0, :], '*')
    legend.append('plan ballx')
    plt.plot(ttraj_val, xtraj_val[1, :], '*')
    legend.append('plan bally')
    plt.plot(ttraj_val, xtraj_val[2, :], '*')
    legend.append('plan paddle')
    # plt.plot(ttraj_val, xtraj_val[5,:],'*')
    # legend.append('plan paddlev')
    plt.plot(ttraj_val_sim, xtraj_sim[0, :])
    legend.append('ballx')
    plt.plot(ttraj_val_sim, xtraj_sim[1, :])
    legend.append('bally')
    plt.plot(ttraj_val_sim, xtraj_sim[2, :])
    legend.append('paddle')
    # plt.plot(ttraj_val_sim, xtraj_sim[5,:])
    # legend.append('paddlev')
    plt.plot(ttraj_val, utraj_val[0, :])
    legend.append('paddle theta')
    plt.legend(legend)
    plt.show()
