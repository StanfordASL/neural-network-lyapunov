import torch
import hybrid_linear_system


class BallPaddleHybridLinearSystem:
    def __init__(self, dtype, dt, x_lo, x_up, u_lo, u_up):
        """
        x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        u = [paddletheta_dot, paddlevy_dot]
        """
        self.dtype = dtype
        self.dt = dt
        self.x_lo = x_lo
        self.x_up = x_up
        self.u_lo = u_lo
        self.u_up = u_up
        
        self.x_dim = 7
        self.u_dim = 2
        
        assert(len(self.x_lo) == self.x_dim)
        assert(len(self.x_up) == self.x_dim)
        assert(len(self.u_lo) == self.u_dim)
        assert(len(self.u_up) == self.u_dim)
        
        # distance from surface considered in collision
        self.collision_eps = 1e-3
        # coefficient of restitution
        self.cr = .8
        
    def get_hybrid_linear_system(self):
        """
        returns a hybrid linear system instance corresponding to 
        the ball paddle system

        x[n+1] = Aᵢ*x[n] + Bᵢ*u[n] + cᵢ
            if Pᵢ * [x[n]; u[n]] <= qᵢ
            i = 1, ..., K.
        """
        g = -9.81
        
        P_xu_lo = -torch.eye(self.x_dim+self.u_dim, dtype=self.dtype)
        P_xu_up = torch.eye(self.x_dim+self.u_dim, dtype=self.dtype)
        P_lim = torch.cat((P_xu_lo, P_xu_up),dim=0)
        q_lim = torch.cat((-self.x_lo,-self.u_lo,self.x_up,self.u_up),dim=0).unsqueeze(1)
        
        hls = hybrid_linear_system.HybridLinearSystem(self.x_dim, self.u_dim, self.dtype)
        
        # free falling mode
        A = torch.Tensor([[1., 0., 0., 0., self.dt, 0., 0.],
                          [0., 1., 0., 0., 0., self.dt, 0.],
                          [0., 0., 1., 0., 0., 0., self.dt],
                          [0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0., 0., 1.]]).type(self.dtype)
        B = torch.Tensor([[0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [self.dt, 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., self.dt]]).type(self.dtype)
        c = torch.Tensor([0., 0., 0., 0., 0., g, 0.]).type(self.dtype).unsqueeze(1)
        # bally > paddley + collision_eps
        P = torch.Tensor([[0., -1., 1., 0., 0., 0., 0., 0., 0.]]).type(self.dtype)
        q = torch.Tensor([-self.collision_eps]).type(self.dtype).unsqueeze(1)
        hls.add_mode(A, B, c, torch.cat((P,P_lim),dim=0), torch.cat((q,q_lim),dim=0))
        
        # colliding with the paddle
        A = torch.Tensor([[1., 0., 0., 0., self.dt, 0., 0.],
                          [0., 0., 1., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 0., 0., self.dt],
                          [0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 0., -self.cr, 1.+self.cr],
                          [0., 0., 0., 0., 0., 0., 1.]]).type(self.dtype)
        B = torch.Tensor([[0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [self.dt, 0.],
                          [0., 0.],
                          [0., self.dt],
                          [0., self.dt]]).type(self.dtype)
        c = torch.Tensor([0., 2.*self.collision_eps, 0., 0., 0., 0., 0.]).type(self.dtype).unsqueeze(1)
        # bally <= paddley + collision_eps
        P = torch.Tensor([[0., 1., -1., 0., 0., 0., 0., 0., 0.]]).type(self.dtype)
        q = torch.Tensor([self.collision_eps]).type(self.dtype).unsqueeze(1)
        hls.add_mode(A, B, c, torch.cat((P,P_lim),dim=0), torch.cat((q,q_lim),dim=0))
      
        return hls