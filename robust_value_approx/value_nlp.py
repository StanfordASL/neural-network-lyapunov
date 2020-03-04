# -*- coding: utf-8 -*-
import robust_value_approx.value_to_optimization as value_to_optimization

import copy
import torch
import numpy as np
import scipy
import jax
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.snopt import SnoptSolver


class NLPValueFunction(value_to_optimization.ValueFunction):
    def __init__(self, x_lo, x_up, u_lo, u_up, dt_lo, dt_up, init_mode=0):
        """
        @param x_lo, x_up, u_lo, u_up are lists of tensors with the lower
        and upper bounds for the states/inputs for each mode
        @param dt_lo, dt_up floats limits for the time step sizes
        """ 
        assert(isinstance(x_lo, list))
        assert(isinstance(x_up, list))
        assert(isinstance(u_lo, list))
        assert(isinstance(u_up, list))
        for bound in [x_lo, x_up, u_lo, u_up]:
            for i in range(len(bound)):
                assert(isinstance(bound[i], torch.Tensor))
        assert(isinstance(dt_lo, float))
        assert(isinstance(dt_up, float))
        assert(len(x_lo) > 0)
        for bound in [x_up, u_lo, u_up]:
            assert(len(x_lo) == len(bound))
        self.x_lo = x_lo
        self.x_up = x_up
        self.u_lo = u_lo
        self.u_up = u_up
        self.x_lo_np = [v.detach().numpy() for v in self.x_lo]
        self.x_up_np = [v.detach().numpy() for v in self.x_up]
        self.u_lo_np = [v.detach().numpy() for v in self.u_lo]
        self.u_up_np = [v.detach().numpy() for v in self.u_up]
        self.dt_lo = dt_lo
        self.dt_up = dt_up
        self.dtype = self.x_lo[0].dtype
        self.x_traj = []
        self.u_traj = []
        self.dt_traj = []
        self.mode_traj = []
        self.num_modes = len(x_lo)
        self.objs = [] # all the objectives in a list
        self.nl_constraints = []
        self.bb_constraints = []
        self.objs_pts = []
        self.nl_constraints_pts = []
        self.bb_constraints_pts = []
        self.objs_dx = []
        self.objs_ddx = []
        self.nl_constraints_dx = []
        self.nl_constraints_ddx = []
        self.nl_constraints_ineq = [] # which con is < vs ==
        self.step_costs = []
        self.pt_index_map_start = []
        self.pt_index_map_end = []
        self.x_dim = [len(x) for x in x_lo]
        self.u_dim = [len(u) for u in u_lo]
        self.prog = MathematicalProgram()
        x0, u0, dt0, mode0 = self.add_knot_point(init_mode)
        self.x0 = x0
        self.u0 = u0
        self.dt0 = dt0
        self.init_mode = mode0
        bb_lo = np.concatenate([np.zeros(self.x_dim[self.init_mode]),
            -np.inf*np.ones(self.u_dim[self.init_mode]), [-np.inf]])
        bb_up = np.concatenate([np.zeros(self.x_dim[self.init_mode]),
            np.inf*np.ones(self.u_dim[self.init_mode]), [np.inf]])
        self.x0_constraint = self.add_bb_constraint(bb_lo, bb_up, [0])
        self.solver = SnoptSolver()

    @property
    def N(self):
        return len(self.x_traj)

    def get_vars(self, pts):
        vars = []
        for n in pts:
            assert(n < self.N)
            x = self.x_traj[n]
            u = self.u_traj[n]
            dt = self.dt_traj[n]
            vars.append(x)
            vars.append(u)
            vars.append(dt)
        return np.concatenate(vars)

    def add_nl_constraint(self, fun, fun_jax, lb, ub, pts):
        con = self.prog.AddConstraint(fun, lb=lb, ub=ub,
            vars=self.get_vars(pts))
        self.nl_constraints.append(con)
        self.nl_constraints_dx.append(jax.jit(jax.jacfwd(fun_jax)))
        self.nl_constraints_ddx.append(
            jax.jit(jax.jacfwd(jax.jacrev(fun_jax))))
        self.nl_constraints_pts.append(pts)
        for i in range(len(lb)):
            if lb[i] != ub[i]:
                self.nl_constraints_ineq.append(1)
            else:
                self.nl_constraints_ineq.append(0)
        return con

    def add_bb_constraint(self, lb, ub, pts):
        con = self.prog.AddBoundingBoxConstraint(lb, ub, self.get_vars(pts))
        self.bb_constraints.append(con)
        self.bb_constraints_pts.append(pts)
        return con

    def add_cost(self, fun, fun_jax, pts):
        obj = self.prog.AddCost(fun, vars=self.get_vars(pts))
        self.objs.append(obj)
        self.objs_dx.append(jax.jit(jax.jacfwd(fun_jax)))
        self.objs_ddx.append(jax.jit(jax.jacfwd(jax.jacrev(fun_jax))))
        self.objs_pts.append(pts)
        return obj

    def get_last_knot_point(self):
        assert(len(self.x_traj) > 0)
        assert(len(self.u_traj) > 0)
        assert(len(self.dt_traj) > 0)
        assert(len(self.mode_traj) > 0)
        return(self.x_traj[-1], self.u_traj[-1], self.dt_traj[-1],
            self.mode_traj[-1])

    def add_knot_point(self, mode):
        x = self.prog.NewContinuousVariables(
            self.x_dim[mode], "x"+str(len(self.x_traj)))
        u = self.prog.NewContinuousVariables(
            self.u_dim[mode], "u"+str(len(self.u_traj)))
        dt = self.prog.NewContinuousVariables(1, "dt"+str(len(self.dt_traj)))
        self.x_traj.append(x)
        self.u_traj.append(u)
        self.dt_traj.append(dt)
        self.mode_traj.append(mode)
        bb_lo = np.concatenate(
            [self.x_lo_np[mode], self.u_lo_np[mode], [self.dt_lo]])
        bb_up = np.concatenate(
            [self.x_up_np[mode], self.u_up_np[mode], [self.dt_up]])
        self.add_bb_constraint(bb_lo, bb_up, [self.N-1])
        self.step_costs.append([])
        if len(self.pt_index_map_end) == 0:
            self.pt_index_map_start.append(0)
        else:
            self.pt_index_map_start.append(self.pt_index_map_end[-1])
        varlen = self.x_dim[mode] + self.u_dim[mode] + 1
        self.pt_index_map_end.append(self.pt_index_map_start[-1] + varlen)
        return(x, u, dt, mode)

    def add_transition(self, transition_fun, transition_fun_jax,
                       guard, guard_jax, new_mode):
        """
        add a knot point and a mode transition to that knot point
        @param transition_fun function that equals 0 for valid transition
        @param guard function that equals 0 at the transition
        @param new_mode index of the resulting mode
        """
        n0 = self.N-1
        n1 = n0+1
        self.add_knot_point(new_mode)
        self.add_nl_constraint(transition_fun, transition_fun_jax,
            np.zeros(self.x_dim[mode1]),
            np.zeros(self.x_dim[mode1]),
            [n0, n1])
        self.add_nl_constraint(guard, guard_jax,
            np.array([0.]),
            np.array([0.]),
            [n0])

    def add_segment(self, N, dyn_fun, dyn_fun_jax,
                    guard=None, guard_jax=None):
        """
        adds a sequence of N knot points in a given mode
        @param dyn_fun function that equals zero for valid transition
        @param guard function that must be positive for the entire mode
        @param N number of knot points
        """
        for n in range(N):
            n0 = self.N-1
            mode0 = self.mode_traj[-1]
            n1 = n0+1
            self.add_knot_point(mode0)
            self.add_nl_constraint(dyn_fun, dyn_fun_jax,
                np.zeros(self.x_dim[mode0]),
                np.zeros(self.x_dim[mode0]),
                [n0, n1])
            if guard is not None:
                assert(guard_jax is not None)
                self.add_nl_constraint(guard, guard_jax,
                    np.array([0.]),
                    np.array([np.inf]),
                    [n1])

    def add_step_cost(self, n, fun, fun_jax):
        assert(n < self.N)
        self.step_costs[n].append(fun)
        self.add_cost(fun, fun_jax, [n])

    def step_cost(self, n, x, u, dt):
        """
        only uses cost that were added using add_step_cost, i.e. that
        are dependent solely on this time step
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        if isinstance(u, torch.Tensor):
            u = u.detach().numpy()
        if isinstance(dt, torch.Tensor):
            dt = dt.detach().numpy()
        if isinstance(dt, float):
            dt = np.array([dt])
        cost = np.sum(
            [c(np.concatenate([x, u, dt])) for c in self.step_costs[n]])
        return cost

    def result_to_costtogo(self, result):
        """
        only uses cost that were added using add_step_cost, i.e. that
        are dependent solely on ONE time step (think additive cost)
        """
        N = self.N
        assert(len(result['x_traj']) == N)
        cost_to_go = torch.zeros(self.N, dtype=self.dtype)
        for n in range(N):
            x = result['x_traj'][N-1-n]
            u = result['u_traj'][N-1-n]
            dt = result['dt_traj'][N-1-n]
            c = self.step_cost(N-1-n, x, u, dt)
            if n > 0:
                c += cost_to_go[N-1-n+1]
            cost_to_go[N-1-n] = c 
        return cost_to_go[:-1]

    def get_value_function(self):
        # alpha solution is always the same for a fixed mode sequence
        alpha_traj_sol = torch.zeros(
            (self.num_modes, self.N), dtype=self.dtype)
        for i in range(len(self.mode_traj)):
            alpha_traj_sol[self.mode_traj[i], i] = 1.
        def V(x):
            assert(isinstance(x, torch.Tensor))
            dtype = x.dtype
            x = x.detach().numpy()
            bb_lo = np.concatenate(
                [x, -np.inf*np.ones(self.u_dim[self.init_mode]), [-np.inf]])
            bb_up = np.concatenate(
                [x, np.inf*np.ones(self.u_dim[self.init_mode]), [np.inf]])
            self.x0_constraint.evaluator().set_bounds(bb_lo, bb_up)
            res = self.solver.Solve(
                self.prog, np.zeros(self.prog.num_vars()), None)
            if not res.is_success():
                return(None, None)
            v_val = res.get_optimal_cost()
            x_traj_sol = [torch.Tensor(
                res.GetSolution(x)).type(self.dtype) for x in self.x_traj]
            u_traj_sol = [torch.Tensor(
                res.GetSolution(u)).type(self.dtype) for u in self.u_traj]
            dt_traj_sol = [res.GetSolution(dt) for dt in self.dt_traj]
            dets = res.get_solver_details()
            result = dict(
                v=v_val,
                x_traj=x_traj_sol,
                u_traj=u_traj_sol,
                dt_traj=torch.Tensor(dt_traj_sol).type(self.dtype),
                alpha_traj=alpha_traj_sol,
                mode_traj=torch.Tensor(self.mode_traj).type(self.dtype),
                fmul=torch.Tensor(dets.Fmul[1:]).type(self.dtype),
                xmul=torch.Tensor(dets.xmul).type(self.dtype),
                )
            return(v_val, result)
        return V

    def pts_val_from_result(self, result, pts):
        var = []
        indices = []
        for n in pts:
            var.append(result['x_traj'][n])
            var.append(result['u_traj'][n])
            var.append([result['dt_traj'][n]])
            indices.append(range(self.pt_index_map_start[n],
                self.pt_index_map_end[n]))
        return(np.concatenate(var), np.concatenate(indices))

    def dfdx(self, result):
        df = np.zeros(self.prog.num_vars())
        for k in range(len(self.objs_dx)):
            var, ind = self.pts_val_from_result(result, self.objs_pts[k])
            df[ind] += self.objs_dx[k](var)
        return df

    def ddfddx(self, result):
        ddf = np.zeros((self.prog.num_vars(), self.prog.num_vars()))
        for k in range(len(self.objs_ddx)):
            var, ind = self.pts_val_from_result(result, self.objs_pts[k])
            x, y = np.meshgrid(ind, ind)
            # not the hessian is symmetric
            ddf[x, y] += self.objs_ddx[k](var)
        return ddf

    def dgdx(self, result):
        # assumes SNOPT doesn't change the order of constraints (it shouldn't)
        dg = []
        for k in range(len(self.nl_constraints_dx)):
            var, ind = self.pts_val_from_result(
                result, self.nl_constraints_pts[k])
            dgi_ = self.nl_constraints_dx[k](var)
            dgi = np.zeros((dgi_.shape[0], self.prog.num_vars()))
            dgi[:, ind] = dgi_
            dg.append(dgi)
        return np.concatenate(dg, axis=0)

    def ddgddx(self, result):
        # assumes SNOPT doesn't change the order of constraints (it shouldn't)
        ddg = []
        for k in range(len(self.nl_constraints_ddx)):
            var, ind = self.pts_val_from_result(
                result, self.nl_constraints_pts[k])
            x, y = np.meshgrid(ind, ind)
            ddgi_ = self.nl_constraints_ddx[k](var)
            ddgi = np.zeros((ddgi_.shape[0], self.prog.num_vars(),
                self.prog.num_vars()))
            # note ther hessian is symmetric wrt last 2 dims
            ddgi[:, x, y] = ddgi_
            ddg.append(ddgi)
        return np.concatenate(ddg, axis=0)

    def dbbdx(self, result):
        eps_tol = 1e-4
        xmul = result['xmul'].detach().numpy()
        xmul[np.abs(xmul) < eps_tol] = 0.
        return np.diag(-np.sign(xmul))

    def get_differentiable_value_function(self):
        V = self.get_value_function()
        V_with_grad = lambda x: DiffFiniteHorizonNLPValueFunction.apply(
            x, self, V)
        return V_with_grad


class DiffFiniteHorizonNLPValueFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vf, V):
        """
        Computes the value function for a finite horizon value function
        that is expressed as a NLP.
        """
        (v, res) = V(x)
        if v is None:
            ctx.success = False
            return(None, None)
        ctx.success = True
        ctx.x = x
        ctx.vf = vf
        ctx.V = V
        ctx.result = res
        x_traj_flat = torch.cat(res['x_traj'])
        cost_to_go = vf.result_to_costtogo(res)
        return(cost_to_go, x_traj_flat)

    @staticmethod
    def backward(ctx, grad_output_cost_to_go, grad_output_x_traj_flat):
        """
        for now only supports one gradient
        """
        assert(torch.all(grad_output_cost_to_go[1:] == 0.))
        assert(torch.all(grad_output_x_traj_flat == 0.))
        if not ctx.success:
            grad_input = torch.zeros(
                ctx.x.shape, dtype=ctx.x.dtype)*float('nan')
            return (grad_input, None, None)
        vf = ctx.vf
        dtype = ctx.x.dtype
        result = ctx.result
        p_dim = vf.x_dim[vf.init_mode]
        dfdx = vf.dfdx(result)
        ddfddx = vf.ddfddx(result)
        dgdx = vf.dgdx(result)
        dbbdx = vf.dbbdx(result)
        dcondx = np.concatenate((dgdx, dbbdx), axis=0)
        ddgddx = vf.ddgddx(result)
        dgdp = np.zeros((dgdx.shape[0], p_dim))
        dbbdp = np.zeros((dbbdx.shape[0], p_dim))        
        dbbdp[:p_dim, :] = np.eye(p_dim)
        dcondp = np.concatenate((dgdp, dbbdp), axis=0)
        Qtl = ddfddx + ddgddx.T@result['fmul'].detach().numpy()
        lhs_top = np.concatenate((Qtl, dcondx.T), axis=1)
        lhs_bottom = np.concatenate(
            (dcondx, np.zeros((dcondx.shape[0], dcondx.shape[0]))), axis=1)
        lhs = np.concatenate((lhs_top, lhs_bottom), axis=0)
        rhs = np.concatenate(
            (np.zeros((vf.prog.num_vars(), p_dim)), dcondp), axis=0)
        lhs = torch.Tensor(lhs).type(dtype)
        rhs = torch.Tensor(rhs).type(dtype)        
        z = torch.inverse(lhs + 1e-12*torch.eye(lhs.shape[0]))@rhs
        dxdp = z[:vf.prog.num_vars(), :]
        dfdp = torch.Tensor(dfdx).type(dtype)@dxdp
        grad_input = (grad_output_cost_to_go[0] * dfdp.unsqueeze(0)).squeeze()
        return(grad_input, None, None)