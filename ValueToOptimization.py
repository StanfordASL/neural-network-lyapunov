# -*- coding: utf-8 -*-
import torch


class ValueFunction:

    def __init__(self, sys):
        """
        Class to store the a value function that can be expressed as a
        Mixed-integer quadratic program.
        
        @param sys: The dynamical system used by the value function
        """
        self.sys = sys
        self.dtype = sys.dtype
        
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
        
        self.xtraj = None
        self.utraj = None
        self.alphatraj = None
        
    def set_cost(self, Q=None, R=None, Z=None, q=None, r=None, z=None):
        """
        Sets the parameters of the additive cost function (not including terminal state)
        
        ∑(.5 (x[n]-xtraj[n])ᵀ Q (x[n]-xtraj[n]) + .5 (u[n]-utraj[n])ᵀ R (u[n]-utraj[n]) + .5 (α[n]-αtraj[n])ᵀ Z (α[n]-αtraj[n]) + qᵀ(x[n]-xtraj[n]) + rᵀ(u[n]-utraj[n]) + zᵀ(α[n]-αtraj[n]))
        
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
        
        .5 (x[N]-xtraj[N])ᵀ Qt (x[N]-xtraj[N]) + .5 (u[N]-utraj[N])ᵀ Rt (u[N]-utraj[N]) + .5 (α[N]-αtraj[N])ᵀ Zt (α[N]-αtraj[N]) + qtᵀ(x[N]-xtraj[N]) + rtᵀ(u[N]-utraj[N]) + ztᵀ(α[N]-αtraj[N])
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

    def set_traj(self, xtraj=None, utraj=None, alphatraj=None):
        """
        Sets the desired trajectory (see description of set_cost and set_terminal_cost).
        
        @param xtraj the desired x trajectory as a statedim by N tensor
        @param utraj the desired u trajectory as a inputdim by N tensor
        @param alphatraj the desired x trajectory as a numdiscretestates by N tensor
        """
        if type(xtraj) != type(None):
            self.xtraj = xtraj.type(self.dtype)
        if type(utraj) != type(None):
            self.utraj = utraj.type(self.dtype)
        if type(alphatraj) != type(None):
            self.alphatraj = alphatraj.type(self.dtype)

    def traj_opt_constraint(self, N):
        """
        Generates a trajectory optimization problem corresponding to the set constraints
        and objectives
        
        min ∑(.5 (x[n]-xtraj[n])ᵀ Q (x[n]-xtraj[n]) + .5 (u[n]-utraj[n])ᵀ R (u[n]-utraj[n]) + .5 (α[n]-αtraj[n])ᵀ Z (α[n]-αtraj[n]) + qᵀ(x[n]-xtraj[n]) + rᵀ(u[n]-utraj[n]) + zᵀ(α[n]-αtraj[n])) + .5 (x[N]-xtraj[N])ᵀ Qt (x[N]-xtraj[N]) + .5 (u[N]-utraj[N])ᵀ Rt (u[N]-utraj[N]) + .5 (α[N]-αtraj[N])ᵀ Zt (α[N]-αtraj[N]) + qtᵀ(x[N]-xtraj[N]) + rtᵀ(u[N]-utraj[N]) + ztᵀ(α[N]-αtraj[N])
        Ain1 x[n] + Ain2 u[n] + Ain3 x[n+1] + Ain4 u[n+1] + Ain5 α[n] ≤ rhs_in_dyn
        Aeq1 x[n] + Aeq2 u[n] + Aeq3 x[n+1] + Aeq4 u[n+1] + Aeq5 α[n] = rhs_eq_dyn
        x_lo ≤ x[n] ≤ x_up
        u_lo ≤ u[n] ≤ u_up
        x[0] == x0
        x[N] == xN
        α[N] ∈ {0,1}
                
        the problem is returned in our standard MIQP form so that it can easily be passed to verification functions.
        Letting x = x[0], and s = x[1]...x[N]
        
        min .5 sᵀ Q2 s + .5 αᵀ Q3 α + q2ᵀ s + q3ᵀ α + c
        s.t. Ain1 x + Ain2 s + Ain3 α ≤ rhs_in
             Aeq1 x + Aeq2 s + Aeq3 α = rhs_eq
             α ∈ {0,1} (needs to be imposed externally)
             
        @return Ain1, Ain2, Ain3, rhs_eq, Aeq1, Aeq2, Aeq3, rhs_eq, Q2, Q3, q2, q3, c
        """
        if type(self.xtraj) != type(None):
            assert(self.xtraj.shape[1] == N-1)
        if type(self.utraj) != type(None):
            assert(self.utraj.shape[1] == N)
        if type(self.alphatraj) != type(None):
            assert(self.alphatraj.shape[1] == N)
        
        # TODO: this needs to be rewritten with HybridLinearSystem api
        Ain1_dyn, Ain2_dyn, Ain3_dyn, Ain4_dyn, Ain5_dyn, rhs_in_dyn = self.sys.get_dyn_in()
        Aeq1_dyn, Aeq2_dyn, Aeq3_dyn, Aeq4_dyn, Aeq5_dyn, rhs_eq_dyn = self.sys.get_dyn_eq()
        
        xdim = Ain1_dyn.shape[1]
        udim = Ain2_dyn.shape[1]
        sdim = (xdim+udim)*N - xdim
        adim = Ain5_dyn.shape[1]
        alphadim = adim*N
        
        # dynamics inequality constraints
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
        Q2 = torch.zeros(sdim,sdim,dtype=self.dtype)
        q2 = torch.zeros(sdim,dtype=self.dtype)
        Q3 = torch.zeros(alphadim,alphadim,dtype=self.dtype)
        q3 = torch.zeros(alphadim,dtype=self.dtype)
        c = 0.
        if type(self.R) != type(None):
            Q2[:udim,:udim] += self.R
            if type(self.utraj) != type(None):
                q2[:udim] -= self.utraj[:,0].T@self.R
                c += .5*self.utraj[:,0].T@self.R@self.utraj[:,0]
        if type(self.r) != type(None):
            q2[:udim] += self.r
            if type(self.utraj) != type(None):
                c -= self.r.T@self.utraj[:,0]
        for i in range(N-2):
            Qi = udim+i*(xdim+udim)
            Qip = udim+i*(xdim+udim)+xdim
            Ri = udim+xdim+i*(xdim+udim)
            Rip = udim+xdim+i*(xdim+udim)+udim
            if type(self.Q) != type(None):
                Q2[Qi:Qip,Qi:Qip] += self.Q
                if type(self.xtraj) != type(None):
                    q2[Qi:Qip] -= self.xtraj[:,i].T@self.Q
                    c += .5*self.xtraj[:,i].T@self.Q@self.xtraj[:,i]
            if type(self.R) != type(None):
                Q2[Ri:Rip,Ri:Rip] += self.R
                if type(self.utraj) != type(None):
                    q2[Ri:Rip] -= self.utraj[:,i+1].T@self.R
                    c += .5*self.utraj[:,i+1].T@self.R@self.utraj[:,i+1]
            if type(self.q) != type(None):
                q2[Qi:Qip] += self.q
                if type(self.xtraj) != type(None):
                    c -= self.q.T@self.xtraj[:,i]
            if type(self.r) != type(None):            
                q2[Ri:Rip] += self.r
                if type(self.utraj) != type(None):
                    c -= self.r.T@self.utraj[:,i+1]
        for i in range(N-1):
            if type(self.Z) != type(None):
                Q3[i*adim:(i+1)*adim,i*adim:(i+1)*adim] += self.Z
                if type(self.alphatraj) != type(None):
                    q3[i*adim:(i+1)*adim] -= self.alphatraj[:,i].T@self.Z
                    c += .5*self.alphatraj[:,i].T@self.Z@self.alphatraj[:,i]                
            if type(self.z) != type(None):
                q3[i*adim:(i+1)*adim] += self.z
                if type(self.alphatraj) != type(None):
                    c -= self.z.T@self.alphatraj[:,i]
        
        if type(self.Qt) != type(None):
            Q2[-(xdim+udim):-udim,-(xdim+udim):-udim] += self.Qt
            if type(self.xtraj) != type(None):
                q2[-(xdim+udim):-udim] -= self.xtraj[:,-1].T@self.Qt
                c += .5*self.xtraj[:,-1].T@self.Qt@self.xtraj[:,-1]
        if type(self.Rt) != type(None):
            Q2[-udim:,-udim:] += self.Rt
            if type(self.utraj) != type(None):
                q2[-udim:] -= self.utraj[:,-1].T@self.Rt
                c += .5*self.utraj[:,-1].T@self.Rt@self.utraj[:,-1]
        if type(self.qt) != type(None):
            q2[-(xdim+udim):-udim] += self.qt
            if type(self.xtraj) != type(None):
                c -= self.qt.T@self.xtraj[:,-1]
        if type(self.rt) != type(None):
            q2[-udim:] += self.rt
            if type(self.utraj) != type(None):
                c -= self.rt.T@self.utraj[:,-1]
        if type(self.Zt) != type(None):
            Q3[-adim:,-adim:] += self.Zt
            if type(self.alphatraj) != type(None):
                q3[-adim:] -= self.alphatraj[:,-1].T@self.Zt
                c += .5*self.alphatraj[:,-1].T@self.Zt@self.alphatraj[:,-1]
        if type(self.zt) != type(None):
            q3[-adim:] += self.zt
            if type(self.alphatraj) != type(None):
                c -= self.zt.T@self.alphatraj[:,-1]
                
        # state and input constraints
        # x_lo ≤ x[n] ≤ x_up
        # u_lo ≤ u[n] ≤ u_up
        if type(self.x_lo) == type(None):
            self.x_lo = torch.ones(xdim,dtype=self.dtype) * -float("Inf")
        if type(self.x_up) == type(None):
            self.x_up = torch.ones(xdim,dtype=self.dtype) * float("Inf")
        if type(self.u_lo) == type(None):
            self.u_lo = torch.ones(udim,dtype=self.dtype) * -float("Inf")
        if type(self.u_up) == type(None):
            self.u_up = torch.ones(udim,dtype=self.dtype) * float("Inf")
        Aup = torch.eye(N*(xdim+udim),N*(xdim+udim),dtype=self.dtype)
        rhs_up = torch.cat((self.x_up,self.u_up)).repeat(N)
        Alo = -torch.eye(N*(xdim+udim),N*(xdim+udim),dtype=self.dtype)
        rhs_lo = -torch.cat((self.x_lo,self.u_lo)).repeat(N)
        Ain1 = torch.cat((Ain1,Aup[:,:xdim],Alo[:,:xdim]),0)
        Ain2 = torch.cat((Ain2,Aup[:,xdim:],Alo[:,xdim:]),0)
        Ain3 = torch.cat((Ain3,torch.zeros(2*N*(xdim+udim),alphadim,dtype=self.dtype)),0)
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

        return(Ain1, Ain2, Ain3, rhs_in, Aeq1, Aeq2, Aeq3, rhs_eq, Q2, Q3, q2, q3, c)