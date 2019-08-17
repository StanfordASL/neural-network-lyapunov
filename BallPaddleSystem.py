import torch
import numpy as np
import cvxpy as cp
from ParametrizedQP import ParametrizedQP

def bound_propagation(model, initial_bounds):
    l, u = initial_bounds
    bounds = []
    
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t() 
                  + layer.bias[:,None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t() 
                  + layer.bias[:,None]).t()
        elif isinstance(layer, torch.nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)
            
        bounds.append((l_, u_))
        l,u = l_, u_
    return bounds

class BallPaddleSystem():
    g = -9.81
    
    def __init__(self,dt=.05,paddle_range=[0., 1.],u_max=5.,cr=.8,M=100.):
        self.dt = dt
        self.paddle_range = paddle_range
        self.u_max = u_max
        self.cr = cr
        self.M = M
        
    def add_dynamics_constraints(self,constraints,N,zb,zp,bi,
                                 dt=None,paddle_range=None,u_max=None,cr=None,M=None):
        if dt == None:
            dt = self.dt
        if paddle_range == None:
            paddle_range = self.paddle_range
        if u_max == None:
            u_max = self.u_max
        if cr == None:
            cr = self.cr
        if M == None:
            M = self.M
        
        for n in range(N-1):
            constraints += [
                zp[0,n+1] == zp[0,n] + zp[1,n+1]*dt,
                zb[0,n+1] == zb[0,n] + zb[1,n+1]*dt,

                zb[1,n+1] - zb[1,n] - self.g*dt >= -M*bi[n],
                zb[1,n+1] - zb[1,n] - self.g*dt <= M*bi[n], 
                (zb[1,n+1] - zp[1,n+1]) + cr*(zb[1,n] - zp[1,n]) >= -M*(1-bi[n]),
                (zb[1,n+1] - zp[1,n+1]) + cr*(zb[1,n] - zp[1,n]) <= M*(1-bi[n]),
            ]
        for n in range(N):
            constraints += [
                zp[0,n] <= paddle_range[1],
                zp[0,n] >= paddle_range[0],
                zp[1,n] <= u_max,
                zp[1,n] >= -u_max,
                zb[0,n] - zp[0,n] >= 0,
                zb[0,n] - zp[0,n] <= M*(1-bi[n]),
            ]
        
    def get_trajectory_miqp(self,paddle_x0,ball_x0,ball_xg,N):
        constraints = []

        zp = cp.Variable([2,N])
        zb = cp.Variable([2,N])
        bi = cp.Variable(N, boolean=True)

        self.add_dynamics_constraints(constraints,N,zb,zp,bi)
        constraints += [
            zp[:,0] == paddle_x0,
            zb[:,0] == ball_x0,
            zb[:,N-1] == ball_xg,
        ]
           
        objective = cp.Minimize(cp.sum_squares(zp[1,:]))

        prob = cp.Problem(objective, constraints)

        return (prob,objective,constraints,{"zp": zp, "zb": zb, "bi": bi})
    
    def get_min_time_trajectory_milp(self,paddle_x0,ball_x0,ball_xg,N):
        constraints = []
        
        zp = cp.Variable([2,N])
        zb = cp.Variable([2,N])
        bi = cp.Variable(N, boolean=True)

        self.add_dynamics_constraints(constraints,N,zb,zp,bi)
        constraints += [
            zp[:,0] == paddle_x0,
            zb[:,0] == ball_x0,
        ]

        # goal indicator variable
        gi = cp.Variable(N, boolean=True)
        for n in range(N-1):
            constraints += [
                ball_xg - zb[:,n] <= self.M*(1-gi[n]),
                ball_xg - zb[:,n] >= -self.M*(1-gi[n]),
            ]
        constraints.append(cp.sum(gi) == 1)
        c = 1000.*(np.array(range(N)) + 1.)
        
        objective = cp.Minimize(c * gi + cp.sum_squares(zp[1,:]))
        
        prob = cp.Problem(objective, constraints)
           
        return (prob,objective,constraints,{"zp": zp, "zb": zb, "bi": bi}) 

    def get_adversarial_miqp(self,model,paddle_x0,ball_x0_min,ball_x0_max,ball_xg,N):
        device = next(model.parameters()).device

        initial_bounds = (torch.tensor([ball_x0_min]).to(device), torch.tensor([ball_x0_max]).to(device))
        bounds = bound_propagation(model, initial_bounds)

        constraints = []
        
        zp = cp.Variable([2,N])
        zb = cp.Variable([2,N])
        bi = cp.Variable(N, boolean=True)

        self.add_dynamics_constraints(constraints,N,zb,zp,bi)
        constraints += [
            zp[:,0] == paddle_x0,
            zb[:,0] >= ball_x0_min,
            zb[:,0] <= ball_x0_max,
            zb[:,N-1] == ball_xg,
        ]

        linear_layers = [(layer, bound) for layer, bound in zip(model,bounds) if isinstance(layer, torch.nn.Linear)]
        d = len(linear_layers)-1

        z = ([cp.Variable(layer.in_features) for layer, _ in linear_layers] + 
             [cp.Variable(linear_layers[-1][0].out_features)])
        v = [cp.Variable(layer.out_features, boolean=True) for layer, _ in linear_layers[:-1]]

        W = [layer.weight.detach().cpu().numpy() for layer,_ in linear_layers]
        b = [layer.bias.detach().cpu().numpy() for layer,_ in linear_layers]
        l = [l[0].detach().cpu().numpy() for _, (l,_) in linear_layers]
        u = [u[0].detach().cpu().numpy() for _, (_,u) in linear_layers]
        l0 = initial_bounds[0][0].view(-1).detach().cpu().numpy()
        u0 = initial_bounds[1][0].view(-1).detach().cpu().numpy()

        for i in range(len(linear_layers)-1):
            constraints += [z[i+1] >= W[i] @ z[i] + b[i], 
                            z[i+1] >= 0,
                            cp.multiply(v[i], u[i]) >= z[i+1],
                            W[i] @ z[i] + b[i] >= z[i+1] + cp.multiply((1-v[i]), l[i])]

        constraints += [z[d+1] == W[d] @ z[d] + b[d]]
        constraints += [z[0] >= l0, z[0] <= u0]
        constraints += [z[0] == zb[:,0]]

        objective = cp.Minimize(cp.sum_squares(zp[1,:]) - z[d+1])
        
        prob = cp.Problem(objective, constraints)
        
        return (prob,objective,constraints,{"zp": zp, "zb": zb, "bi": bi, "z": z, "v": v})
    
    def get_adversarial_qp(self,model,paddle_x0,ball_x0_min,ball_x0_max,ball_xg,N,bi=None,v=None):
        device = next(model.parameters()).device

        initial_bounds = (torch.tensor([ball_x0_min]).to(device), torch.tensor([ball_x0_max]).to(device))
        bounds = bound_propagation(model, initial_bounds)

        constraints = []

        zp = cp.Variable([2,N])
        zb = cp.Variable([2,N])
        if bi == None:
            bi = cp.Variable(N)
            constraints += [bi >= 0.,
                            bi <= 1.]
            
        self.add_dynamics_constraints(constraints,N,zb,zp,bi)
        constraints += [
            zp[:,0] == paddle_x0,
            zb[:,0] >= ball_x0_min,
            zb[:,0] <= ball_x0_max,
            zb[:,N-1] == ball_xg,
        ]

        linear_layers = [(layer, bound) for layer, bound in zip(model,bounds) if isinstance(layer, torch.nn.Linear)]
        d = len(linear_layers)-1

        z = ([cp.Variable(layer.in_features) for layer, _ in linear_layers] + 
             [cp.Variable(linear_layers[-1][0].out_features)])
        
        if v == None:
            v = [cp.Variable(layer.out_features) for layer, _ in linear_layers[:-1]]
            constraints += [vi >= 0. for vi in v]
            constraints += [vi <= 1. for vi in v]
        
        W = [layer.weight.detach().cpu().numpy() for layer,_ in linear_layers]
                
        b = [layer.bias.detach().cpu().numpy() for layer,_ in linear_layers]
        l = [l[0].detach().cpu().numpy() for _, (l,_) in linear_layers]
        u = [u[0].detach().cpu().numpy() for _, (_,u) in linear_layers]
        l0 = initial_bounds[0][0].view(-1).detach().cpu().numpy()
        u0 = initial_bounds[1][0].view(-1).detach().cpu().numpy()

        for i in range(len(linear_layers)-1):
            constraints += [z[i+1] >= W[i] @ z[i] + b[i], 
                            z[i+1] >= 0,
                            cp.multiply(v[i], u[i]) >= z[i+1],
                            W[i] @ z[i] + b[i] >= z[i+1] + cp.multiply((1-v[i]), l[i])]

        constraints += [z[d+1] == W[d] @ z[d] + b[d]]
        constraints += [z[0] >= l0, z[0] <= u0]
        
        constraints += [z[0] == zb[:,0]]
        
        objective = cp.Minimize(cp.sum_squares(zp[1,:]) - z[d+1])

        prob = cp.Problem(objective, constraints)
        
        return (prob,objective,constraints,{"zp": zp, "zb": zb, "bi": bi, "z": z, "v": v})
    
    def get_adversarial_qp_standard(self,model,paddle_x0,ball_x0_min,ball_x0_max,ball_xg,N,bi=None,v=None):
        device = next(model.parameters()).device       
        dtype = next(model.parameters()).type()
        
        prob = ParametrizedQP(device, dtype=dtype)
        
        for n in range(N):
            prob.add_var("zp" + str(n), 2)
            prob.add_var("zb" + str(n), 2)
            prob.add_var("bi" + str(n), 1)
            
        for n in range(N-1):
            prob.add_eq(["zp"+str(n),"zp"+str(n+1)],[torch.tensor([1.,0.]),torch.tensor([-1.,self.dt])],torch.tensor([0.]))
            prob.add_eq(["zb"+str(n),"zb"+str(n+1)],[torch.tensor([1.,0.]),torch.tensor([-1.,self.dt])],torch.tensor([0.]))
            
            prob.add_ineq(["zb"+str(n),"bi"+str(n),"zb"+str(n+1)],[torch.tensor([0.,1.]),torch.tensor([-self.M]),torch.tensor([0.,-1.])],torch.tensor([-self.g*self.dt]))
            prob.add_ineq(["zb"+str(n),"bi"+str(n),"zb"+str(n+1)],[torch.tensor([0.,-1.]),torch.tensor([-self.M]),torch.tensor([0.,1.])],torch.tensor([self.g*self.dt]))
            
            prob.add_ineq(["zb"+str(n),"zp"+str(n),"bi"+str(n),"zb"+str(n+1),"zp"+str(n+1)],[torch.tensor([0.,-self.cr]),torch.tensor([0.,self.cr]),torch.tensor([self.M]),torch.tensor([0.,-1.]),torch.tensor([0.,1.])],torch.tensor([self.M]))
            prob.add_ineq(["zb"+str(n),"zp"+str(n),"bi"+str(n),"zb"+str(n+1),"zp"+str(n+1)],[torch.tensor([0.,self.cr]),torch.tensor([0.,-self.cr]),torch.tensor([self.M]),torch.tensor([0.,1.]),torch.tensor([0.,-1.])],torch.tensor([self.M]))
            
        for n in range(N):
            prob.add_ineq(["zp"+str(n)],[torch.tensor([1.,0.])],torch.tensor([self.paddle_range[1]]))
            prob.add_ineq(["zp"+str(n)],[torch.tensor([0.,1.])],torch.tensor([self.u_max]))
            prob.add_ineq(["zp"+str(n)],[torch.tensor([-1.,0.])],torch.tensor([-self.paddle_range[0]]))
            prob.add_ineq(["zp"+str(n)],[torch.tensor([0.,-1.])],torch.tensor([self.u_max]))
            prob.add_ineq(["zb"+str(n),"zp"+str(n)],[torch.tensor([-1.,0.]),torch.tensor([1.,0.])],torch.tensor([0.]))
            prob.add_ineq(["zb"+str(n),"bi"+str(n),"zp"+str(n)],[torch.tensor([1.,0.]),torch.tensor([self.M]),torch.tensor([-1.,0.])],torch.tensor([self.M]))
            
            prob.add_ineq(["bi"+str(n)],[torch.tensor([-1.])],torch.tensor([0.]))
            prob.add_ineq(["bi"+str(n)],[torch.tensor([1.])],torch.tensor([1.]))

        prob.add_eq(["zp0"],[torch.eye(2)],torch.tensor(paddle_x0))
        prob.add_ineq(["zb0"],[-torch.eye(2)],-torch.tensor(ball_x0_min))
        prob.add_ineq(["zb0"],[torch.eye(2)],torch.tensor(ball_x0_max))
        prob.add_eq(["zb"+str(N-1)],[torch.eye(2)],torch.tensor(ball_xg))

        initial_bounds = (torch.tensor([ball_x0_min]).to(device), torch.tensor([ball_x0_max]).to(device))
        bounds = bound_propagation(model, initial_bounds)
        linear_layers = [(layer, bound) for layer, bound in zip(model,bounds) if isinstance(layer, torch.nn.Linear)]
        d = len(linear_layers)-1
        
        for i in range(len(linear_layers)):
            prob.add_var("z"+str(i), linear_layers[i][0].in_features)
        prob.add_var("z"+str(len(linear_layers)), linear_layers[-1][0].out_features)
        for i in range(len(linear_layers)-1):
            prob.add_var("v"+str(i), linear_layers[i][0].out_features)
        
        W = [layer.weight for layer,_ in linear_layers]
        b = [layer.bias for layer,_ in linear_layers]
        l = [l[0] for _, (l,_) in linear_layers]
        u = [u[0] for _, (_,u) in linear_layers]
        l0 = initial_bounds[0][0].view(-1)
        u0 = initial_bounds[1][0].view(-1)
        
        for i in range(len(linear_layers)-1):
            prob.add_ineq(["z"+str(i),"z"+str(i+1)],[W[i],-torch.eye(W[i].shape[0])],-b[i])
            prob.add_ineq(["z"+str(i+1)],[-torch.eye(W[i+1].shape[1])],torch.zeros(W[i+1].shape[1]))
            prob.add_ineq(["z"+str(i+1),"v"+str(i)],[torch.eye(W[i+1].shape[1]),-torch.diag(u[i])],torch.zeros(W[i+1].shape[1]))
            prob.add_ineq(["z"+str(i+1),"z"+str(i),"v"+str(i)],[torch.eye(W[i+1].shape[1]),-W[i],-torch.diag(l[i])],b[i]-l[i])
            prob.add_ineq(["v"+str(i)],[-torch.eye(linear_layers[i][0].out_features)],torch.zeros(linear_layers[i][0].out_features))
            prob.add_ineq(["v"+str(i)],[torch.eye(linear_layers[i][0].out_features)],torch.ones(linear_layers[i][0].out_features))
            
        prob.add_eq(["z"+str(d+1),"z"+str(d)],[torch.eye(1),-W[d]],b[d])
        prob.add_ineq(["z0"],[-torch.eye(W[0].shape[1])],-l0)
        prob.add_ineq(["z0"],[torch.eye(W[0].shape[1])],u0)
                
        prob.add_eq(["z0","zb0"],[torch.eye(W[0].shape[1]),-torch.eye(2)],torch.zeros(2))
        
        for n in range(N):
            prob.set_obj("zp"+str(n),quadratic=torch.tensor([[0.,0.],[0.,1.]]))
        prob.set_obj("z"+str(d+1),linear=-torch.ones(1))
        
        if type(bi) != type(None):
            for n in range(N):
                prob.add_eq(["bi"+str(n)],[torch.ones(1)],torch.tensor([bi[n]]))
        if type(v) != type(None):
            for i in range(d):
                prob.add_eq(["v"+str(i)],[torch.eye(linear_layers[i][0].out_features)],torch.tensor(v[i]))
        
        return prob