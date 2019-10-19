# -*- coding: utf-8 -*-
import torch
from utils import torch_to_numpy
import cvxpy as cp


class ValueFunction:

    def __init__(self, sys, N):
        """
        Class to store the a value function that can be expressed as a
        Mixed-integer quadratic program.
        
        @param sys: The dynamical system used by the value function
        """
        self.sys = sys
        self.dtype = sys.dtype
        self.N = N
        
        self.Q = None
        self.R = None
        self.Z = None
        self.q = None
        self.r = None
        self.z = None
        self.Qt = None
        self.Rt = None
        self.Zt = None
        self.qt = None
        self.rt = None
        self.zt = None
        
        self.x_lo = None
        self.x_up = None
        self.u_lo = None
        self.u_up = None
        self.x0 = None
        self.xN = None
        
    def set_cost(self, Q=None, R=None, Z=None, q=None, r=None, z=None):
        """
        Sets the parameters of the additive cost function (not including terminal state)
        
        ∑(.5 xᵀ[n] Q x[n] + .5 uᵀ[n] R u[n] + .5 αᵀ[n] Z α[n] + qᵀx[n] + rᵀu[n] + zᵀα[n])
        
        for n = 0...N-1
        """
        if type(Q) != type(None):
            self.Q = Q.type(self.dtype)
        if type(R) != type(None):
            self.R = R.type(self.dtype)
        if type(Z) != type(None):
            self.Z = Z.type(self.dtype)
        if type(q) != type(None):
            self.q = q.type(self.dtype)
        if type(r) != type(None):
            self.r = r.type(self.dtype)
        if type(z) != type(None):
            self.z = z.type(self.dtype)
        
    def set_terminal_cost(self, Qt=None, Rt=None, Zt=None, qt=None, rt=None, zt=None):
        """
        Set the parameters of the terminal cost
        
        .5 xᵀ[N] Qt x[N] + .5 uᵀ[N] Rt u[N] + .5 αᵀ[N] Zt α[N] + qtᵀx[N] + rtᵀu[N] + ztᵀα[N]
        """
        if type(Qt) != type(None):
            self.Qt = Qt.type(self.dtype)
        if type(Rt) != type(None):
            self.Rt = Rt.type(self.dtype)
        if type(Zt) != type(None):
            self.Zt = Zt.type(self.dtype)
        if type(qt) != type(None):
            self.qt = qt.type(self.dtype)
        if type(rt) != type(None):
            self.rt = rt.type(self.dtype)
        if type(zt) != type(None):
            self.zt = zt.type(self.dtype)

    def set_constraints(self, x_lo=None, x_up=None, u_lo=None, u_up=None, x0=None, xN=None):
        """ 
        Sets the constraints for the optimization (imposed on every state along the trajectory)
        
        x_lo ≤ x[n] ≤ x_up
        u_lo ≤ u[n] ≤ u_up
        x[0] == x0
        x[N] == xN
        """
        if type(x_lo) != type(None):
            self.x_lo = x_lo.type(self.dtype)
        if type(x_up) != type(None):
            self.x_up = x_up.type(self.dtype)
        if type(u_lo) != type(None):
            self.u_lo = u_lo.type(self.dtype)
        if type(u_up) != type(None):
            self.u_up = u_up.type(self.dtype)
        if type(x0) != type(None):
            self.x0 = x0.type(self.dtype)
        if type(xN) != type(None):
            self.xN = xN.type(self.dtype)

    def traj_opt_constraint(self):
        """
        Generates a trajectory optimization problem corresponding to the set constraints
        and objectives
        
        min ∑(.5 xᵀ[n] Q x[n] + .5 uᵀ[n] R u[n] + .5 αᵀ[n] Z α[n] + qᵀx[n] + rᵀu[n] + zᵀα[n])
                + .5 xᵀ[N] Qt x[N] + .5 uᵀ[N] Rt u[N] + .5 αᵀ[N] Zt α[N] + qtᵀx[N] + rtᵀu[N] + ztᵀα[N]
        Ain1 x[n] + Ain2 u[n] + Ain3 x[n+1] + Ain4 u[n+1] + Ain5 α[n] ≤ rhs_in_dyn
        Aeq1 x[n] + Aeq2 u[n] + Aeq3 x[n+1] + Aeq4 u[n+1] + Aeq5 α[n] = rhs_eq_dyn
        x_lo ≤ x[n] ≤ x_up
        u_lo ≤ u[n] ≤ u_up
        x[0] == x0
        x[N] == xN
        α[N] ∈ {0,1}
                
        the problem is returned in our standard MIQP form so that it can easily be passed to verification functions.
        Letting x = x[0], and s = x[1]...x[N]
        
        min .5 sᵀ Q2 s + .5 αᵀ Q3 α + q2ᵀ s + q3ᵀ α
        s.t. Ain1 x + Ain2 s + Ain3 α ≤ rhs_in
             Aeq1 x + Aeq2 s + Aeq3 α = rhs_eq
             α ∈ {0,1} (needs to be imposed externally)
             
        @return Ain1, Ain2, Ain3, rhs_eq, Aeq1, Aeq2, Aeq3, rhs_eq, Q2, Q3, q2, q3
        """
        # TODO: this needs to be rewritten with HybridLinearSystem api
        Ain1_dyn, Ain2_dyn, Ain3_dyn, Ain4_dyn, Ain5_dyn, rhs_in_dyn = self.sys.get_dyn_in()
        Aeq1_dyn, Aeq2_dyn, Aeq3_dyn, Aeq4_dyn, Aeq5_dyn, rhs_eq_dyn = self.sys.get_dyn_eq()

        N = self.N        
        xdim = Ain1_dyn.shape[1]
        udim = Ain2_dyn.shape[1]
        sdim = (xdim+udim)*N - xdim
        adim = Ain5_dyn.shape[1]
        alphadim = adim*N
        
        # dynamics inequality constraints
        # Ain1 x[n] + Ain2 u[n] + Ain3 x[n+1] + Ain4 u[n+1] + Ain5 α[n] ≤ rhs_in_dyn
        num_in_dyn = rhs_in_dyn.shape[0]
        s_in_dyn = torch.cat((Ain1_dyn, Ain2_dyn, Ain3_dyn, Ain4_dyn), 1)
        Ain = torch.zeros((N-1)*num_in_dyn,N*(xdim+udim),dtype=self.dtype)
        Ain3 = torch.zeros((N-1)*num_in_dyn,alphadim,dtype=self.dtype)
        rhs_in = torch.zeros((N-1)*num_in_dyn,dtype=self.dtype)
        for i in range(N-1):
            Ain[i*num_in_dyn:(i+1)*num_in_dyn,i*(xdim+udim):i*(xdim+udim)+2*(xdim+udim)] = s_in_dyn
            Ain3[i*num_in_dyn:(i+1)*num_in_dyn,i*adim:(i+1)*adim] = Ain5_dyn
            rhs_in[i*num_in_dyn:(i+1)*num_in_dyn] = rhs_in_dyn.squeeze()
        Ain1 = Ain[:,:xdim]
        Ain2 = Ain[:,xdim:]
        
        # dynamics equality constraints
        # Aeq1 x[n] + Aeq2 u[n] + Aeq3 x[n+1] + Aeq4 u[n+1] + Aeq5 α[n] = rhs_eq_dyn
        num_eq_dyn = rhs_eq_dyn.shape[0]
        s_eq_dyn = torch.cat((Aeq1_dyn, Aeq2_dyn, Aeq3_dyn, Aeq4_dyn), 1)
        Aeq = torch.zeros((N-1)*num_eq_dyn,N*(xdim+udim),dtype=self.dtype)
        Aeq3 = torch.zeros((N-1)*num_eq_dyn,alphadim,dtype=self.dtype)
        rhs_eq = torch.zeros((N-1)*num_eq_dyn,dtype=self.dtype)
        for i in range(N-1):
            Aeq[i*num_eq_dyn:(i+1)*num_eq_dyn,i*(xdim+udim):i*(xdim+udim)+2*(xdim+udim)] = s_eq_dyn
            Aeq3[i*num_eq_dyn:(i+1)*num_eq_dyn,i*adim:(i+1)*adim] = Aeq5_dyn
            rhs_eq[i*num_eq_dyn:(i+1)*num_eq_dyn] = rhs_eq_dyn.squeeze()
        Aeq1 = Aeq[:,0:xdim]
        Aeq2 = Aeq[:,xdim:]
        
        # costs 
        # .5 xᵀ[n] Q x[n] + .5 uᵀ[n] R u[n] + .5 αᵀ[n] Z α[n] + qᵀx[n] + rᵀu[n] + zᵀα[n]
        Q2 = torch.zeros(sdim,sdim,dtype=self.dtype)
        q2 = torch.zeros(sdim,dtype=self.dtype)
        Q3 = torch.zeros(alphadim,alphadim,dtype=self.dtype)
        q3 = torch.zeros(alphadim,dtype=self.dtype)
        if type(self.R) != type(None):
            Q2[:udim,:udim] = self.R
        if type(self.r) != type(None):
            q2[:udim] = self.r
        for i in range(N-2):
            Qi = udim+i*(xdim+udim)
            Qip = udim+i*(xdim+udim)+xdim
            Ri = udim+xdim+i*(xdim+udim)
            Rip = udim+xdim+i*(xdim+udim)+udim
            if type(self.Q) != type(None):
                Q2[Qi:Qip,Qi:Qip] = self.Q
            if type(self.R) != type(None):
                Q2[Ri:Rip,Ri:Rip] = self.R
            if type(self.q) != type(None):
                q2[Qi:Qip] = self.q
            if type(self.r) != type(None):            
                q2[Ri:Rip] = self.r
        for i in range(N):
            if type(self.Z) != type(None):
                Q3[i*adim:(i+1)*adim,i*adim:(i+1)*adim] = self.Z
            if type(self.z) != type(None):
                q3[i*adim:(i+1)*adim] = self.z
        
        # .5 xᵀ[N] Qt x[N] + .5 uᵀ[N] Rt u[N] + .5 αᵀ[N] Zt α[N] + qtᵀx[N] + rtᵀu[N] + ztᵀα[N]
        if type(self.Qt) != type(None):
            Q2[-(xdim+udim):-udim,-(xdim+udim):-udim] = self.Qt
        if type(self.Rt) != type(None):
            Q2[-udim:,-udim:] = self.Rt
        if type(self.qt) != type(None):
            q2[-(xdim+udim):-udim] = self.qt
        if type(self.rt) != type(None):
            q2[-udim:] = self.rt
        if type(self.Zt) != type(None):
            Q3[-adim:,-adim:] = self.Zt
        if type(self.zt) != type(None):
            q3[-adim:] = self.zt

        # state and input constraints
        # x_lo ≤ x[n] ≤ x_up
        # u_lo ≤ u[n] ≤ u_up
        if type(self.x_up) != type(None) and type(self.u_up) != type(None):
            Aup = torch.eye(N*(xdim+udim),N*(xdim+udim),dtype=self.dtype)
            rhs_up = torch.cat((self.x_up,self.u_up)).repeat(N)
            Ain3 = torch.cat((Ain3,torch.zeros(N*(xdim+udim),alphadim,dtype=self.dtype)),0)
        elif type(self.x_up) != type(None):
            Aup = torch.eye(N*(xdim+udim),N*(xdim+udim),dtype=self.dtype)
            Aup = Aup[torch.cat((torch.ones(xdim),torch.zeros(udim))).repeat(N).type(torch.bool),:]
            rhs_up = self.x_up.repeat(N)
            Ain3 = torch.cat((Ain3,torch.zeros(N*xdim,alphadim,dtype=self.dtype)),0)
        elif type(self.u_up) != type(None):
            Aup = torch.eye(N*(xdim+udim),N*(xdim+udim),dtype=self.dtype)
            Aup = Aup[torch.cat((torch.zeros(xdim),torch.ones(udim))).repeat(N).type(torch.bool),:]
            rhs_up = self.u_up.repeat(N)
            Ain3 = torch.cat((Ain3,torch.zeros(N*udim,alphadim,dtype=self.dtype)),0)
        else:
            Aup = torch.zeros(0,N*(xdim+udim),dtype=self.dtype)
            rhs_up = torch.zeros(0,dtype=self.dtype)
            
        if type(self.x_lo) != type(None) and type(self.u_lo) != type(None):
            Alo = -torch.eye(N*(xdim+udim),N*(xdim+udim),dtype=self.dtype)
            rhs_lo = -torch.cat((self.x_lo,self.u_lo)).repeat(N)
            Ain3 = torch.cat((Ain3,torch.zeros(N*(xdim+udim),alphadim,dtype=self.dtype)),0)
        elif type(self.x_lo) != type(None):
            Alo = -torch.eye(N*(xdim+udim),N*(xdim+udim),dtype=self.dtype)
            Alo = Alo[torch.cat((torch.ones(xdim),torch.zeros(udim))).repeat(N).type(torch.bool),:]
            rhs_lo = -self.x_lo.repeat(N)
            Ain3 = torch.cat((Ain3,torch.zeros(N*xdim,alphadim,dtype=self.dtype)),0)
        elif type(self.u_lo) != type(None):
            Alo = -torch.eye(N*(xdim+udim),N*(xdim+udim),dtype=self.dtype)
            Alo = Alo[torch.cat((torch.zeros(xdim),torch.ones(udim))).repeat(N).type(torch.bool),:]
            rhs_lo = -self.u_lo.repeat(N)
            Ain3 = torch.cat((Ain3,torch.zeros(N*udim,alphadim,dtype=self.dtype)),0)
        else:
            Alo = torch.zeros(0,N*(xdim+udim),dtype=self.dtype)
            rhs_lo = torch.zeros(0,dtype=self.dtype)
        
        Ain1 = torch.cat((Ain1,Aup[:,:xdim],Alo[:,:xdim]),0)
        Ain2 = torch.cat((Ain2,Aup[:,xdim:],Alo[:,xdim:]),0)
        rhs_in = torch.cat((rhs_in,rhs_up,rhs_lo))
        
        # initial state constraints
        # x[0] == x0
        if type(self.x0) != type(None):
            Aeq1 = torch.cat((Aeq1,torch.eye(xdim,dtype=self.dtype)),0)
            Aeq2 = torch.cat((Aeq2,torch.zeros(xdim,sdim,dtype=self.dtype)),0)
            Aeq3 = torch.cat((Aeq3,torch.zeros(xdim,alphadim,dtype=self.dtype)),0)
            rhs_eq = torch.cat((rhs_eq,self.x0))

        # final state constraint
        # x[N] == xN
        if type(self.xN) != type(None):
            Aeq1 = torch.cat((Aeq1,torch.zeros(xdim,xdim,dtype=self.dtype)),0)
            Aeq2 = torch.cat((Aeq2,torch.zeros(xdim,sdim,dtype=self.dtype)),0)
            Aeq2[-xdim:,-udim-xdim:-udim] = torch.eye(xdim,dtype=self.dtype)
            Aeq3 = torch.cat((Aeq3,torch.zeros(xdim,alphadim,dtype=self.dtype)),0)
            rhs_eq = torch.cat((rhs_eq,self.xN))

        return(Ain1, Ain2, Ain3, rhs_in, Aeq1, Aeq2, Aeq3, rhs_eq, Q2, Q3, q2, q3)
            
    def get_value_function(self):
        """
        return a function that can be evaluated to get the optimal cost-to-go
        for a given initial state. Uses cvxpy in order to solve the cost-to-go
        
        @return V a function handle that takes x0, the initial state as a tensor
        and return the associated optimal cost-to-go as a scalar
        """
        traj_opt = self.traj_opt_constraint()
        (Ain1, Ain2, Ain3, rhs_in, 
        Aeq1, Aeq2, Aeq3, rhs_eq, 
        Q2, Q3, q2, q3) = torch_to_numpy(traj_opt)
        
        s = cp.Variable(Ain2.shape[1])
        alpha = cp.Variable(Ain3.shape[1],boolean=True)
        
        obj = cp.Minimize(.5*cp.quad_form(s,Q2) + .5*cp.quad_form(alpha,Q3) + q2.T@s + q3.T@alpha)    

        def V(x0):
            if type(x0) == torch.Tensor:
                x0 = x0.detach().numpy().squeeze()
            
            con = [
                Ain1@x0 + Ain2@s + Ain3@alpha <= rhs_in,
                Aeq1@x0 + Aeq2@s + Aeq3@alpha == rhs_eq,
            ]        
            
            prob = cp.Problem(obj,con)
            prob.solve(solver=cp.GUROBI, verbose=False)
            
            return(obj.value,s.value,alpha.value)
        
        return V
        