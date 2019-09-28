# -*- coding: utf-8 -*-
import torch


class ValueFunction:

    def __init__(self,sys):
        """
        Class to store the a value function that can be expressed as a
        Mixed-integer quadratic program.
        
        @param sys The dynamical system used by the value function
        """
        self.sys = sys

    def traj_opt_x0xN(self, Q, R, Z, q, r, z, 
                      Qt, Rt, Zt, qt, rt, zt,
                      N, x0, xN,
                      x_lo, x_up, u_lo, u_up):
        """
        Generates a trajectory optimization problem that is constrained by 
        its initial and final state (x0 and xN). The arguments correspond 
        to the following problem formulation
        
        min ∑(.5 xᵀ[n] Q x[n] + .5 uᵀ[n] R u[n] + .5 αᵀ[n] Z α[n] + qᵀx[n] + rᵀu[n] + zᵀα[n])
                + .5 xᵀ[N] Qt x[N] + .5 uᵀ[N] Rt u[N] + .5 αᵀ[N] Zt α[N] + qtᵀx[N] + rtᵀu[N] + ztᵀα[N]
        Ain1 x[n] + Ain2 u[n] + Ain3 x[n+1] + Ain4 u[n+1] + Ain5 α[n] ≤ rhs_in_dyn
        Aeq1 x[n] + Aeq2 u[n] + Aeq3 x[n+1] + Aeq4 u[n+1] + Aeq5 α[n] = rhs_eq_dyn
        x_lo ≤ x[n] ≤ x_up (optional)
        u_lo ≤ u[n] ≤ u_up (optional)
        x[0] == x0
        x[N] == xs
        α[N] ∈ {0,1}
                
        the problem is returned in our standard MIQP form so that it can easily be passed to verification functions.
        Letting x = x[0], and s = x[1]...x[N]
        
        min .5 sᵀ Q2 s + .5 αᵀ Q3 α + q2ᵀ s + q3ᵀ α
        s.t. Ain1 x + Ain2 s + Ain3 α ≤ brhs_in
             Aeq1 x + Aeq2 s + Aeq3 α = brhs_eq
             α ∈ {0,1} (needs to be imposed externally)
        """
        sys = self.sys
        Ain1_dyn, Ain2_dyn, Ain3_dyn, Ain4_dyn, Ain5_dyn, rhs_in_dyn = sys.get_dyn_in()
        Aeq1_dyn, Aeq2_dyn, Aeq3_dyn, Aeq4_dyn, Aeq5_dyn, rhs_eq_dyn = sys.get_dyn_eq()
        
        xdim = Ain1_dyn.shape[1]
        udim = Ain2_dyn.shape[1]
        sdim = (xdim+udim)*N - xdim
        adim = Ain5_dyn.shape[1]
        alphadim = adim*N
        
        Q2 = torch.zeros(sdim,sdim,dtype=sys.dtype)
        q2 = torch.zeros(sdim,dtype=sys.dtype)
        Q3 = torch.zeros(alphadim,alphadim,dtype=sys.dtype)
        q3 = torch.zeros(alphadim,dtype=sys.dtype)

        # .5 xᵀ[n] Q x[n] + .5 uᵀ[n] R u[n] + .5 αᵀ[n] Z α[n] + qᵀx[n] + rᵀu[n] + zᵀα[n]
        Q2[:udim,:udim] = R
        q2[:udim] = r
        for i in range(N-2):
            Qi = udim+i*(xdim+udim)
            Qip = udim+i*(xdim+udim)+xdim
            Ri = udim+xdim+i*(xdim+udim)
            Rip = udim+xdim+i*(xdim+udim)+udim
            Q2[Qi:Qip,Qi:Qip] = Q
            Q2[Ri:Rip,Ri:Rip] = R
            q2[Qi:Qip] = q
            q2[Ri:Rip] = r
        for i in range(N):
            Q3[i*adim:(i+1)*adim,i*adim:(i+1)*adim] = Z
            q3[i*adim:(i+1)*adim] = z
        
        # .5 xᵀ[N] Qt x[N] + .5 uᵀ[N] Rt u[N] + .5 αᵀ[N] Zt α[N] + qtᵀx[N] + rtᵀu[N] + ztᵀα[N]
        Q2[-(xdim+udim):-udim,-(xdim+udim):-udim] = Qt
        Q2[-udim:,-udim:] = Rt
        q2[-(xdim+udim):-udim] = qt
        q2[-udim:] = rt
        Q3[-adim:,-adim:] = Zt
        q3[-adim:] = zt
        
        # Ain1 x[n] + Ain2 u[n] + Ain3 x[n+1] + Ain4 u[n+1] + Ain5 α[n] ≤ rhs_in_dyn
        num_in_dyn = rhs_in_dyn.shape[0]
        s_in_dyn = torch.cat((Ain1_dyn, Ain2_dyn, Ain3_dyn, Ain4_dyn), 1)
        Ain = torch.zeros((N-1)*num_in_dyn,N*(xdim+udim),dtype=sys.dtype)
        Ain3 = torch.zeros((N-1)*num_in_dyn,alphadim,dtype=sys.dtype)
        brhs_in = torch.zeros((N-1)*num_in_dyn,dtype=sys.dtype)
        for i in range(N-1):
            Ain[i*num_in_dyn:(i+1)*num_in_dyn,i*(xdim+udim):i*(xdim+udim)+2*(xdim+udim)] = s_in_dyn
            Ain3[i*num_in_dyn:(i+1)*num_in_dyn,i*adim:(i+1)*adim] = Ain5_dyn
            brhs_in[i*num_in_dyn:(i+1)*num_in_dyn] = rhs_in_dyn.squeeze()
        Ain1 = Ain[:,:xdim]
        Ain2 = Ain[:,xdim:]
        
        # Aeq1 x[n] + Aeq2 u[n] + Aeq3 x[n+1] + Aeq4 u[n+1] + Aeq5 α[n] = rhs_eq_dyn
        num_eq_dyn = rhs_eq_dyn.shape[0]
        s_eq_dyn = torch.cat((Aeq1_dyn, Aeq2_dyn, Aeq3_dyn, Aeq4_dyn), 1)
        Aeq = torch.zeros((N-1)*num_eq_dyn,N*(xdim+udim),dtype=sys.dtype)
        Aeq3 = torch.zeros((N-1)*num_eq_dyn,alphadim,dtype=sys.dtype)
        brhs_eq = torch.zeros((N-1)*num_eq_dyn,dtype=sys.dtype)
        for i in range(N-1):
            Aeq[i*num_eq_dyn:(i+1)*num_eq_dyn,i*(xdim+udim):i*(xdim+udim)+2*(xdim+udim)] = s_eq_dyn
            Aeq3[i*num_eq_dyn:(i+1)*num_eq_dyn,i*adim:(i+1)*adim] = Aeq5_dyn
            brhs_eq[i*num_eq_dyn:(i+1)*num_eq_dyn] = rhs_eq_dyn.squeeze()
        Aeq1 = Aeq[:,0:xdim]
        Aeq2 = Aeq[:,xdim:]

        # x_lo ≤ x[n] ≤ x_up
        # u_lo ≤ u[n] ≤ u_up
        Aup = torch.eye(N*(xdim+udim),N*(xdim+udim),dtype=sys.dtype)
        brhs_up = torch.cat((x_up,u_up)).repeat(N)
        Alo = -torch.eye(N*(xdim+udim),N*(xdim+udim),dtype=sys.dtype)
        brhs_lo = -torch.cat((x_lo,u_lo)).repeat(N)
        
        Ain1 = torch.cat((Ain1,Aup[:,:xdim],Alo[:,:xdim]),0)
        Ain2 = torch.cat((Ain2,Aup[:,xdim:],Alo[:,xdim:]),0)
        Ain3 = torch.cat((Ain3,torch.zeros(2*N*(xdim+udim),alphadim,dtype=sys.dtype)),0)
        brhs_in = torch.cat((brhs_in,brhs_up,brhs_lo))
        
        # x[0] == x0
        Aeq1 = torch.cat((Aeq1,torch.eye(xdim,dtype=sys.dtype)),0)
        Aeq2 = torch.cat((Aeq2,torch.zeros(xdim,sdim,dtype=sys.dtype)),0)
        Aeq3 = torch.cat((Aeq3,torch.zeros(xdim,alphadim,dtype=sys.dtype)),0)
        brhs_eq = torch.cat((brhs_eq,x0))

        # x[N] == xN
        Aeq1 = torch.cat((Aeq1,torch.zeros(xdim,xdim,dtype=sys.dtype)),0)
        Aeq2 = torch.cat((Aeq2,torch.zeros(xdim,sdim,dtype=sys.dtype)),0)
        Aeq2[-xdim:,-udim-xdim:-udim] = torch.eye(xdim,dtype=sys.dtype)
        Aeq3 = torch.cat((Aeq3,torch.zeros(xdim,alphadim,dtype=sys.dtype)),0)
        brhs_eq = torch.cat((brhs_eq,xN))

        return(Q2, Q3, q2, q3, Ain1, Ain2, Ain3, brhs_in, Aeq1, Aeq2, Aeq3, brhs_eq)