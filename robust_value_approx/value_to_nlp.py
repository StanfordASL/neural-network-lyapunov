from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.snopt import SnoptSolver

import numpy as np
import matplotlib.pyplot as plt


class NLPValueFunction():
    def __init__(self, x_lo, x_up, u_lo, u_up,
                 dt_lo=0., dt_up=.1, init_mode=0):
        """
        @param x_lo, x_up, u_lo, u_up are lists of tensors with the lower
        and upper bounds for the states/inputs for each mode
        @param dt_lo, dt_up floats limits for the time step sizes
        @param init_mode integer mode the system initializes in
        """ 
        assert(isinstance(x_lo, list))
        assert(isinstance(x_up, list))
        assert(isinstance(u_lo, list))
        assert(isinstance(u_up, list))
        self.x_lo = x_lo
        self.x_up = x_up
        self.u_lo = u_lo
        self.u_up = u_up
        self.dt_lo = dt_lo
        self.dt_up = dt_up
        self.x_traj = []
        self.u_traj = []
        self.dt_traj = []
        self.mode_traj = []
        self.x_dim = [len(x) for x in x_lo]
        self.u_dim = [len(u) for u in u_lo]
        self.prog = MathematicalProgram()
        x0, u0, dt0, mode0 = self.add_knot_point(init_mode)
        self.x0_constraint = self.prog.AddBoundingBoxConstraint(
            np.zeros(self.x_dim[mode0]),
            np.zeros(self.x_dim[mode0]),
            x0)
        self.solver = SnoptSolver()

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
        self.prog.AddBoundingBoxConstraint(self.x_lo[mode], self.x_up[mode], x)
        self.prog.AddBoundingBoxConstraint(self.u_lo[mode], self.u_up[mode], u)
        self.prog.AddBoundingBoxConstraint(self.dt_lo, self.dt_up, dt)
        return(x, u, dt, mode)

    def add_transition(self, transition_fun, guard, new_mode):
        """
        add a knot point and a mode transition to that knot point
        @param transition_fun function that equals 0 for valid transition
        @param guard function that equals 0 at the transition
        @param new_mode index of the resulting mode
        """
        x0, u0, dt0, mode0 = self.get_last_knot_point()
        x1, u1, dt1, mode1 = self.add_knot_point(new_mode)
        self.prog.AddConstraint(transition_fun,
            lb=np.zeros(self.x_dim[mode1]),
            ub=np.zeros(self.x_dim[mode1]),
            vars=np.concatenate([x0, u0, dt0, x1, u1, dt1]))
        self.prog.AddConstraint(guard,
            lb=np.array([0.]),
            ub=np.array([0.]),
            vars=np.concatenate([x0, u0, dt0]))

    def add_mode(self, dyn_fun, guard, N):
        """
        adds a mode for N knot points
        @param dyn_fun function that equals zero for valid transition
        @param guard function that must be positive for the entire mode
        @param N number of knot points
        """
        for n in range(N):
            x0, u0, dt0, mode0 = self.get_last_knot_point()
            x1, u1, dt1, mode1 = self.add_knot_point(mode0)
            self.prog.AddConstraint(dyn_fun,
                lb=np.zeros(self.x_dim[mode0]),
                ub=np.zeros(self.x_dim[mode0]),
                vars=np.concatenate([x0, u0, dt0, x1, u1, dt1]))
            self.prog.AddConstraint(guard,
                lb=np.array([0.]),
                ub=np.array([np.inf]),
                vars=np.concatenate([x1, u1, dt1]))

    def add_terminal_cost(self, Q, x_desired):
        """
        adds quadratic cost to the last state
        """
        x0, u0, dt0, mode0 = self.get_last_knot_point()
        self.prog.AddQuadraticErrorCost(Q=Q, x_desired=x_desired, vars=x0)

    def get_value_function(self):
        def V(x):
            self.x0_constraint.evaluator().set_bounds(x, x)
            result = self.solver.Solve(
                self.prog, np.random.rand(self.prog.num_vars()), None)
            print(result.get_solution_result())
            return result
        return V

    def result_to_traj(self, result):
        x_traj_sol = [result.GetSolution(x) for x in self.x_traj]
        u_traj_sol = [result.GetSolution(u) for u in self.u_traj]
        dt_traj_sol = [result.GetSolution(dt) for dt in self.dt_traj]
        mode_traj_sol = [mode for mode in self.mode_traj]
        return(x_traj_sol, u_traj_sol, dt_traj_sol, mode_traj_sol)


class SLIPNLP:
    def __init__(self):
        self.mass = 80
        self.l0 = 1
        self.gravity = 9.81
        self.dimensionless_spring_constant = 10.3
        self.k = self.dimensionless_spring_constant * self.mass *\
            self.gravity / self.l0
        self.x_lo = [-1e4*np.ones(4),
                     np.array([0., -np.pi/2, -1e6, -1e6, -1e6])]
        self.x_up = [1e4*np.ones(4),
                     np.array([self.l0, np.pi/2, 1e6, 1e6, 1e6])]
        self.u_lo = [np.array([-np.pi/2]), np.array([0])]
        self.u_up = [np.array([np.pi/2]), np.array([0])]
        # self.Qt = np.diag([0., 10., 1., 1.])
        self.Qt = np.diag([10., 1., 0., 0.])

    def flight_dyn(self, var):
        x_dim = 4
        u_dim = 1
        x0 = var[:x_dim] 
        u0 = var[x_dim:x_dim+u_dim]
        dt0 = var[x_dim+u_dim]
        x1 = var[x_dim+u_dim+1:x_dim+u_dim+1+x_dim]
        dx0 = np.array([x0[2], x0[3], 0, -self.gravity])
        return x0 + dt0 * dx0 - x1

    def touchdown_guard(self, var):
        x_dim = 4
        u_dim = 1
        x0 = var[:x_dim] 
        u0 = var[x_dim:x_dim+u_dim]
        dt0 = var[x_dim+u_dim]
        y = x0[1]
        theta = u0[0]
        return np.array([y - self.l0 * np.cos(theta)])

    def flight_to_stance(self, var):
        x_dim0 = 4
        u_dim0 = 1
        x_dim1 = 5
        x0 = var[:x_dim0]
        u0 = var[x_dim0:x_dim0+u_dim0]
        dt0 = var[x_dim0+u_dim0:x_dim0+u_dim0+1]
        x1 = var[x_dim0+u_dim0+1:x_dim0+u_dim0+1+x_dim1]
        theta = u0[0]
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        xdot = x0[2]
        ydot = x0[3]
        x_next = np.array([self.l0, theta,
                           -xdot * sin_theta + ydot * cos_theta,
                           (-xdot * cos_theta - ydot * sin_theta) / self.l0,
                           x0[0] + self.l0 * sin_theta])
        return x1 - x_next

    def stance_dyn(self, var):
        x_dim = 5
        u_dim = 1
        x0 = var[:x_dim]
        u0 = var[x_dim:x_dim+u_dim]
        dt0 = var[x_dim+u_dim:x_dim+u_dim+1]
        x1 = var[x_dim+u_dim+1:x_dim+u_dim+1+x_dim]
        r = x0[0]
        theta = x0[1]
        r_dot = x0[2]
        theta_dot = x0[3]
        r_ddot = r * theta_dot ** 2 - self.gravity * np.cos(theta) + self.k * (self.l0 - r) / self.mass
        theta_ddot = self.gravity / r * np.sin(theta) - 2 * r_dot * theta_dot / r
        dx0 = np.array([r_dot, theta_dot, r_ddot, theta_ddot, 0])
        return x0 + dt0 * dx0 - x1

    def liftoff_guard(self, var):
        x_dim = 5
        x0 = var[:x_dim]
        r = x0[0]
        return np.array([self.l0 - r])

    def stance_to_flight(self, var):
        x_dim0 = 5
        u_dim0 = 1
        x_dim1 = 4
        x0 = var[:x_dim0]
        u0 = var[x_dim0:x_dim0+u_dim0]
        dt0 = var[x_dim0+u_dim0]
        x1 = var[x_dim0+u_dim0+1:x_dim0+u_dim0+1+x_dim1]
        r = x0[0]
        theta = x0[1]
        r_dot = x0[2]
        theta_dot = x0[3]
        x_foot = x0[4]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_next = np.array([x_foot - self.l0 * sin_theta,
                           self.l0 * cos_theta,
                           -r * cos_theta * theta_dot - r_dot * sin_theta,
                           -r * sin_theta * theta_dot + r_dot * cos_theta])
        return x1 - x_next

    def get_nlp_value_function(self, xf, num_transitions, N):
        vf = NLPValueFunction(self.x_lo, self.x_up,
            self.u_lo, self.u_up, init_mode=0)
        for n in range(num_transitions):
            vf.add_mode(self.flight_dyn, self.touchdown_guard, N)
            vf.add_transition(self.flight_to_stance, self.touchdown_guard, 1)
            vf.add_mode(self.stance_dyn, self.liftoff_guard, N)
            vf.add_transition(self.stance_to_flight, self.liftoff_guard, 0)
        vf.add_mode(self.flight_dyn, self.touchdown_guard, N)
        vf.add_terminal_cost(self.Qt, xf)
        return vf

    def plot_traj(self, x_traj, mode_traj=None):
        pos = []
        for x in x_traj:
            if len(x) == 4:
                pos.append([x[0], x[1]]) 
            elif len(x) == 5:
                r = x[0]
                theta = x[1]
                x_foot = x[4]
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                pos.append([x_foot - r * sin_theta, r * cos_theta])
        pos_traj = np.array(pos).T
        plt.plot(pos_traj[0,:], pos_traj[1,:])
        if mode_traj is not None:
            plt.plot(pos_traj[0,:], mode_traj)
        plt.show()


def main():
    slip = SLIPNLP()
    x0 = np.array([0., 1.25, 3., 0.])
    xf = np.array([10., 1., 3., 0.])
    vf = slip.get_nlp_value_function(xf, 2, 10)
    V = vf.get_value_function()
    res = V(x0)
    (x_traj_sol, u_traj_sol, dt_traj_sol, mode_traj_sol) = vf.result_to_traj(res)
    slip.plot_traj(x_traj_sol, mode_traj_sol)


if __name__=="__main__":
    main()