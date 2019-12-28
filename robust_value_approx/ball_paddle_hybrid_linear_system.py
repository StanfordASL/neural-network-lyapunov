import torch
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.constants as constants


def get_ball_paddle_hybrid_linear_system(dtype, dt, x_lo, x_up, u_lo, u_up,
                                         paddle_angles=None,
                                         ball_capture_lo=None,
                                         ball_capture_up=None,
                                         cr=.8, collision_eps=1e-4,
                                         midpoint=False):
    """
    x = [ballx, bally, paddley, ballvx, ballvy, paddlevy]
    u = [paddletheta, paddlevy_dot]
    @param paddle_angles A tensor of the different angles the paddle can take
    and the system will be linearized around
    @param ball_capture_lo A tensor of the lower bound for a bounding
    box where the state becomes stationary (e.g. reaching the goal).
    The dynamics are ALLOWED (not forced) to go to zero if
    ball_capture_lo <= [ballx, bally] <= ball_capture_up
    @param ball_capture_up see ball_capture_lo
    @param cr The coefficient of restitution
    @param collision_eps The distance at which we consider collision
    @param midpoint Whether or not to use midpoint integration
    """
    if paddle_angles is None:
        paddle_angles = torch.Tensor([0.]).type(dtype)
    x_dim = 6
    u_dim = 2
    assert(len(x_lo) == x_dim)
    assert(len(x_up) == x_dim)
    assert(len(u_lo) == u_dim)
    assert(len(u_up) == u_dim)
    P_xu_lo = -torch.eye(x_dim + u_dim, dtype=dtype)
    P_xu_up = torch.eye(x_dim + u_dim, dtype=dtype)
    P_lim = torch.cat((P_xu_lo, P_xu_up), dim=0)
    q_lim = torch.cat((-x_lo, -u_lo, x_up, u_up), dim=0)
    hls = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)
    if midpoint:
        X = torch.Tensor([[1., 0., 0., -.5*dt, 0., 0.],
                          [0., 1., 0., 0., -.5*dt, 0.],
                          [0., 0., 1., 0., 0., -.5*dt],
                          [0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0., 1.]]).type(dtype)
        Xinv = torch.inverse(X)
        A = Xinv @ torch.Tensor([[1., 0., 0., .5*dt, 0., 0.],
                                 [0., 1., 0., 0., .5*dt, 0.],
                                 [0., 0., 1., 0., 0., .5*dt],
                                 [0., 0., 0., 1., 0., 0.],
                                 [0., 0., 0., 0., 1., 0.],
                                 [0., 0., 0., 0., 0., 1.]]).type(dtype)
        B = Xinv @ torch.Tensor([[0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., dt]]).type(dtype)
        c = Xinv @ torch.Tensor([0., 0., 0.,
                                 0., dt*constants.G, 0.]).type(dtype)
    else:
        A = torch.Tensor([[1., 0., 0., dt, 0., 0.],
                          [0., 1., 0., 0., dt, 0.],
                          [0., 0., 1., 0., 0., dt],
                          [0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0., 1.]]).type(dtype)
        B = torch.Tensor([[0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., dt]]).type(dtype)
        c = torch.Tensor([0., 0., 0., 0., dt*constants.G, 0.]).type(dtype)
    # free falling mode away from paddle
    # ballx > 0
    P = torch.Tensor([[-1., 0., 0., 0., 0., 0., 0., 0.]]).type(dtype)
    q = torch.Tensor([0.]).type(dtype)
    hls.add_mode(A, B, c, torch.cat(
        (P, P_lim), dim=0), torch.cat(
            (q, q_lim), dim=0))
    # free falling mode over paddle
    # bally > paddley + collision_eps
    P = torch.Tensor([[0., -1., 1., 0., -dt, dt, 0., 0.]]).type(dtype)
    q = torch.Tensor([-collision_eps]).type(dtype)
    hls.add_mode(A, B, c, torch.cat(
        (P, P_lim), dim=0), torch.cat(
            (q, q_lim), dim=0))
    for theta in paddle_angles:
        st = torch.sin(theta)
        ct = torch.cos(theta)
        if midpoint:
            X = torch.Tensor([[1., 0., 0., -.5*dt, 0., 0.],
                              [0., 1., 0., 0., 0., -.5*dt],
                              [0., 0., 1., 0., 0., -.5*dt],
                              [0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 0., 1., 0.],
                              [0., 0., 0., 0., 0., 1.]]).type(dtype)
            Xinv = torch.inverse(X)
            A = Xinv @ torch.Tensor([[1., 0., 0., .5*dt, 0., 0.],
                                     [0., 0., 1., 0., 0., .5*dt],
                                     [0., 0., 1., 0., 0., .5*dt],
                                     [0., 0., 0., 1., -st*cr, st*cr],
                                     [0., 0., 0., 0., -ct*cr, 1. + ct*cr],
                                     [0., 0., 0., 0., 0., 1.]]).type(dtype)
            B = Xinv @ torch.Tensor([[0., 0.],
                                     [0., 0.],
                                     [0., 0.],
                                     [0., 0.],
                                     [0., 0.],
                                     [0., dt]]).type(dtype)
            c = Xinv @ torch.Tensor([0., collision_eps,
                                     0., 0., 0., 0.]).type(dtype)
        else:
            A = torch.Tensor([[1., 0., 0., dt, 0., 0.],
                              [0., 0., 1., 0., 0., dt],
                              [0., 0., 1., 0., 0., dt],
                              [0., 0., 0., 1., -st*cr, st*cr],
                              [0., 0., 0., 0., -ct*cr, 1. + ct*cr],
                              [0., 0., 0., 0., 0., 1.]]).type(dtype)
            B = torch.Tensor([[0., 0.],
                              [0., 0.],
                              [0., 0.],
                              [0., 0.],
                              [0., 0.],
                              [0., dt]]).type(dtype)
            c = torch.Tensor([0., collision_eps, 0., 0., 0., 0.]).type(dtype)
        # colliding with the paddle
        # bally <= paddley + collision_eps
        # 0 <= ballx <= 0
        # theta <= paddle angle <= theta
        P = torch.Tensor([[0., 1., -1., 0., dt, -dt, 0., 0.],
                          [1., 0., 0., 0., 0., 0., 0., 0.],
                          [-1., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0., 0., -1., 0.]]).type(dtype)
        q = torch.Tensor([collision_eps, 0., 0., theta, -theta]).type(dtype)
        hls.add_mode(A, B, c, torch.cat(
            (P, P_lim), dim=0), torch.cat(
                (q, q_lim), dim=0))
    if ball_capture_lo is not None or ball_capture_up is not None:
        assert(ball_capture_lo is not None)
        assert(ball_capture_up is not None)
        A = torch.Tensor([[1., 0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.]]).type(dtype)
        B = torch.Tensor([[0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., 0.]]).type(dtype)
        c = torch.Tensor([0., 0., 0., 0., 0., 0.]).type(dtype)
        # capture state
        # goal_x_lo <= ball_x <= goal_x_up
        # goal_y_lo <= ball_y <= goal_y_up
        P = torch.Tensor([[-1., 0., 0., 0., 0., 0., 0., 0.],
                          [0., -1., 0., 0., 0., 0., 0., 0.],
                          [1., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0., 0., 0., 0.]]).type(dtype)
        q = torch.Tensor([-ball_capture_lo[0],
                          -ball_capture_lo[1],
                          ball_capture_up[0],
                          ball_capture_up[1]]).type(dtype)
        hls.add_mode(A, B, c, torch.cat(
            (P, P_lim), dim=0), torch.cat(
                (q, q_lim), dim=0))
    return hls
