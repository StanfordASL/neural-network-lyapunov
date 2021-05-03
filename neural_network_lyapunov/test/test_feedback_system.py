import torch
import gurobipy

import numpy as np

import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.compute_xhat as compute_xhat

import unittest


class TestFeedbackSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        controller_network_params1 = torch.tensor([
            0.2, 1.2, 0.4, -0.1, 2.5, 1.3, 0.3, 0.1, 0.5, 0.3, 1.2, -0.5, 0.2,
            0.3, -0.5, 0.1, 1.2, -2.1, 0.3, 1.2, 0.5, 0.1, -0.5, 1.2, 0.1, 1.2,
            -1.3, 0.3, 0.1, -0.8, 0.3, -0.5
        ],
                                                  dtype=self.dtype)
        self.controller_network1 = utils.setup_relu((3, 3, 3, 2),
                                                    controller_network_params1,
                                                    negative_slope=0.01,
                                                    bias=True,
                                                    dtype=self.dtype)

        controller_network_params2 = torch.tensor([
            0.1, 0.4, 0.2, 0.5, -0.3, -1.2, 0.5, 0.9, 0.8, 0.7, -2.1, 0.4, 0.5,
            0.1, 1.2, 1.5, 0.4, -0.3, -2.1, -1.4, -0.9, 0.45, 0.32, 0.12, 0.78,
            -0.5
        ],
                                                  dtype=self.dtype)
        self.controller_network2 = utils.setup_relu((4, 3, 2, 1),
                                                    controller_network_params2,
                                                    negative_slope=0.01,
                                                    bias=True,
                                                    dtype=self.dtype)

        # A simple linear controller
        self.controller_network3 = torch.nn.Linear(4, 1)
        self.controller_network3.weight.data = torch.tensor(
            [[0.5, 0.6, 0.7, 0.8]], dtype=self.dtype)
        self.controller_network3.bias.data = torch.tensor([1.5],
                                                          dtype=self.dtype)

    def add_dynamics_mip_constraint_tester(self, forward_system,
                                           controller_network, x_equilibrium,
                                           u_equilibrium, x_val, u_lower_limit,
                                           u_upper_limit):
        # This function checks if add_dynamics_mip_constraint imposes the right
        # constraint, such that x_next and u computed from MIP matches with
        # that computed from the controller network and the foward dynamics.
        dut = feedback_system.FeedbackSystem(forward_system,
                                             controller_network, x_equilibrium,
                                             u_equilibrium, u_lower_limit,
                                             u_upper_limit)
        milp = gurobi_torch_mip.GurobiTorchMILP(dtype=self.dtype)
        x = milp.addVars(dut.x_dim,
                         lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS,
                         name="x")
        x_next = milp.addVars(dut.x_dim,
                              lb=-gurobipy.GRB.INFINITY,
                              vtype=gurobipy.GRB.CONTINUOUS,
                              name="x_next")
        u, forward_slack, controller_slack, forward_binary, controller_binary\
            = dut.add_dynamics_mip_constraint(
                milp, x, x_next, "u", "forward_s", "forward_binary",
                "controller_s", "controller_binary")

        milp.addMConstrs([torch.eye(forward_system.x_dim, dtype=self.dtype)],
                         [x],
                         sense=gurobipy.GRB.EQUAL,
                         b=x_val)

        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, 0)
        milp.gurobi_model.optimize()

        u_val = torch.tensor([ui.X for ui in u], dtype=self.dtype)
        u_val_expected = controller_network(x_val) -\
            controller_network(x_equilibrium) + u_equilibrium
        u_val_expected = torch.max(
            torch.min(u_val_expected, torch.from_numpy(u_upper_limit)),
            torch.from_numpy(u_lower_limit))
        np.testing.assert_allclose(u_val.detach().numpy(),
                                   u_val_expected.detach().numpy())

        x_next_val = torch.tensor([xi.X for xi in x_next], dtype=self.dtype)
        x_next_expected = dut.step_forward(x_val)
        np.testing.assert_allclose(x_next_val.detach().numpy(),
                                   x_next_expected.detach().numpy())

    def construct_hybrid_linear_system_example(self):
        forward_system = hybrid_linear_system.HybridLinearSystem(
            3, 2, self.dtype)
        # x[n+1] = x[n] + [u[n]; 0] + [1;0;0] if [0, -1, -1] <= x[n] <= [1,1,1]
        # and [-10, -10] <= u[n] <= [10, 10]
        # x[n+1] = x[n] + [0; u[n]] + [1;0;0] if [-1,-1,-1] <= x[n] <= [0,1,1]
        # and [-10, -10] <= u[n] <= [10, 10]
        P1 = torch.zeros(10, 5, dtype=self.dtype)
        P1[:3, :3] = torch.eye(3, dtype=self.dtype)
        P1[3:6, :3] = -torch.eye(3, dtype=self.dtype)
        P1[6:8, 3:] = torch.eye(2, dtype=self.dtype)
        P1[8:, 3:] = -torch.eye(2, dtype=self.dtype)
        forward_system.add_mode(
            torch.eye(3, dtype=self.dtype),
            torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=self.dtype),
            torch.tensor([1, 0, 0], dtype=self.dtype), P1,
            torch.tensor([1, 1, 1, 0, 1, 1, 10, 10, 10, 10], dtype=self.dtype))
        forward_system.add_mode(
            torch.eye(3, dtype=self.dtype),
            torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=self.dtype),
            torch.tensor([1, 0, 0], dtype=self.dtype), P1,
            torch.tensor([0, 1, 1, 1, 1, 1, 10, 10, 10, 10], dtype=self.dtype))
        return forward_system

    def test_add_dynamics_mip_constraint_hybrid_linear_system(self):
        forward_system = self.construct_hybrid_linear_system_example()
        # Our network should work even if the constraint x* = f(x*, u*) is not
        # satisfied at the equilibrium.
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network1, x_equilibrium,
            u_equilibrium, torch.tensor([0.1, 0.2, 0.3], dtype=self.dtype),
            np.array([-1, -2.]), np.array([1, 2.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network1, x_equilibrium,
            u_equilibrium, torch.tensor([-0.1, 0.2, 0.3], dtype=self.dtype),
            np.array([0, 0.]), np.array([100, 100.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network1, x_equilibrium,
            u_equilibrium, torch.tensor([0.1, 0.0, 0.0], dtype=self.dtype),
            np.array([-100., -100.]), np.array([0.2, 0.3]))

    def construct_relu_forward_system(self):
        forward_network_params = torch.tensor([
            0.1, 0.4, 0.5, 0.5, 0.1, 0.2, -0.1, 1.2, 1.1, -1.2, 0.5, -0.3, 0.2,
            0.1, 0.4, 1.1, 0.4, -0.5, 0.1, 0.3, 0.2, 0.2, -0.5, -0.9, 0.8, 1.5,
            0.3, 0.3, 0.5, 0.1
        ],
                                              dtype=self.dtype)
        forward_network = utils.setup_relu((5, 3, 3),
                                           forward_network_params,
                                           negative_slope=0.01,
                                           bias=True,
                                           dtype=self.dtype)
        forward_system = relu_system.ReLUSystem(
            self.dtype, torch.tensor([-2, -2, -2], dtype=self.dtype),
            torch.tensor([2, 2, 2], dtype=self.dtype),
            torch.tensor([-5, -5], dtype=self.dtype),
            torch.tensor([5, 5], dtype=self.dtype), forward_network)
        return forward_system

    def test_add_dynamics_mip_constraint_relu_system(self):
        # Construct a relu network as the forward dynamical system.
        forward_system = self.construct_relu_forward_system()

        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network1, x_equilibrium,
            u_equilibrium, torch.tensor([0.1, 0.2, 0.3], dtype=self.dtype),
            np.array([-1, 0.]), np.array([1., 1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network1, x_equilibrium,
            u_equilibrium, torch.tensor([-0.1, 0.2, 0.3], dtype=self.dtype),
            np.array([-10, -20.]), np.array([1, 1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network1, x_equilibrium,
            u_equilibrium, torch.tensor([0.1, 0.0, 0.0], dtype=self.dtype),
            np.array([0., 0.]), np.array([10., 20.]))

    def test_strengthen_controller_mip_constraint(self):
        # This is a behavior test. We only make sure the strengthened
        # constraint gives tighter constraint, but we don't check if the
        # coefficients/bounds of the strengthened constraint is correct.
        x_equilibrium = torch.tensor([0.5, 0.2, 0.1], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.2, -0.3], dtype=self.dtype)
        forward_system = self.construct_relu_forward_system_given_equilibrium(
            x_equilibrium, u_equilibrium)
        milp = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
        x_var = milp.addVars(forward_system.x_dim, lb=-gurobipy.GRB.INFINITY)
        u_var = milp.addVars(forward_system.u_dim, lb=-gurobipy.GRB.INFINITY)
        dut = feedback_system.FeedbackSystem(
            forward_system, self.controller_network1, x_equilibrium,
            u_equilibrium,
            forward_system.u_lo.detach().numpy(),
            forward_system.u_up.detach().numpy())
        mip_cnstr_return = dut._add_controller_mip_constraint(
            milp, x_var, u_var, "s", "b", lp_relaxation=True)
        # Now solve this MIP with an arbitrary cost.
        milp.setObjective(
            [torch.ones((forward_system.u_dim, ), dtype=self.dtype)], [u_var],
            constant=0.,
            sense=gurobipy.GRB.MAXIMIZE)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.optimize()
        self.assertEqual(milp.gurobi_model.status, gurobipy.GRB.Status.OPTIMAL)
        obj_val = milp.gurobi_model.ObjVal
        # This solution should be non-integral.
        binary_sol = np.array([v.x for v in mip_cnstr_return.binary])
        assert (np.logical_or(np.any(np.abs(binary_sol) > 1E-6),
                              np.any(np.abs(binary_sol - 1) > 1E-6)))
        num_ineq = len(milp.rhs_in)
        dut.strengthen_controller_mip_constraint(milp, x_var,
                                                 mip_cnstr_return.slack,
                                                 mip_cnstr_return.binary,
                                                 mip_cnstr_return.post_relu_lo,
                                                 mip_cnstr_return.post_relu_up)
        # Check if new inequality constraints are added.
        self.assertGreater(len(milp.rhs_in), num_ineq)
        milp.gurobi_model.optimize()
        # Make sure we get smaller objective with the strengthened constraint.
        self.assertGreater(obj_val, milp.gurobi_model.ObjVal)

    def construct_relu_forward_system_given_equilibrium(
            self, x_equilibrium, u_equilibrium):
        forward_network_params = torch.tensor([
            0.1, 0.4, 0.5, 0.5, 0.1, 0.2, -0.1, 1.2, 1.1, -1.2, 0.5, -0.3, 0.2,
            0.1, 0.4, 1.1, 0.4, -0.5, 0.1, 0.3, 0.2, 0.2, -0.5, -0.9, 0.8, 1.5,
            0.3, 0.3, 0.5, 0.1
        ],
                                              dtype=self.dtype)
        forward_network = utils.setup_relu((5, 3, 3),
                                           forward_network_params,
                                           negative_slope=0.01,
                                           bias=True,
                                           dtype=self.dtype)
        forward_system = relu_system.ReLUSystemGivenEquilibrium(
            self.dtype, torch.tensor([-2, -2, -2], dtype=self.dtype),
            torch.tensor([2, 2, 2], dtype=self.dtype),
            torch.tensor([-5, -5], dtype=self.dtype),
            torch.tensor([5, 5], dtype=self.dtype), forward_network,
            x_equilibrium, u_equilibrium)
        return forward_system

    def construct_relu_second_order_forward_system_given_equilibrium(
            self, q_equilibrium, u_equilibrium, dt):
        # A second order system with nq = nv = 2, nu = 1
        forward_network_params = torch.tensor([
            0.2, 0.4, 0.1, 0.5, 0.4, -0.2, 0.4, 0.5, 0.9, -0.3, 1.2, -2.1, 0.1,
            0.45, 0.2, 0.8, 0.7, 0.3, 0.2, 0.5, 0.4, 0.8, 2.1, 0.4, 0.5, 0.2,
            -0.4, -0.5, 0.3, -2.1, 0.4, 0.2, 0.1, 0.5
        ],
                                              dtype=self.dtype)
        forward_network = utils.setup_relu((5, 4, 2),
                                           forward_network_params,
                                           negative_slope=0.01,
                                           bias=True,
                                           dtype=self.dtype)
        forward_system = relu_system.ReLUSecondOrderSystemGivenEquilibrium(
            self.dtype, torch.tensor([-2, -2, -4, -4], dtype=self.dtype),
            torch.tensor([2, 2, 4, 4], dtype=self.dtype),
            torch.tensor([-10], dtype=self.dtype),
            torch.tensor([10], dtype=self.dtype), forward_network,
            q_equilibrium, u_equilibrium, dt)
        return forward_system

    def construct_relu_second_order_residue_system_given_equilibrium(self):
        # Construct a ReLU system with nq = 2, nv = 2 and nu = 1
        self.dtype = torch.float64
        dynamics_relu = utils.setup_relu((3, 5, 2),
                                         params=None,
                                         negative_slope=0.01,
                                         bias=True,
                                         dtype=self.dtype)
        dynamics_relu[0].weight.data = torch.tensor(
            [[0.1, 0.2, 0.3], [0.5, -0.2, 0.4], [0.1, 0.3, -1.2],
             [1.5, 0.3, 0.3], [0.2, 1.5, 0.1]],
            dtype=self.dtype)
        dynamics_relu[0].bias.data = torch.tensor([0.1, -1.2, 0.3, 0.2, -0.5],
                                                  dtype=self.dtype)
        dynamics_relu[2].weight.data = torch.tensor(
            [[0.1, -2.3, 1.5, 0.4, 0.2], [0.1, -1.2, -1.3, 0.3, 0.8]],
            dtype=self.dtype)
        dynamics_relu[2].bias.data = torch.tensor([0.2, -1.4],
                                                  dtype=self.dtype)

        x_lo = torch.tensor([-2, -2, -5, -5], dtype=self.dtype)
        x_up = torch.tensor([2, 2, 5, 5], dtype=self.dtype)
        u_lo = torch.tensor([-5], dtype=self.dtype)
        u_up = torch.tensor([5], dtype=self.dtype)
        q_equilibrium = torch.tensor([0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.4], dtype=self.dtype)
        dt = 0.01
        forward_system = relu_system.\
            ReLUSecondOrderResidueSystemGivenEquilibrium(
                self.dtype, x_lo, x_up, u_lo, u_up, dynamics_relu,
                q_equilibrium, u_equilibrium, dt,
                network_input_x_indices=[1, 3])
        return forward_system

    def test_add_dynamics_mip_constraint_relu_second_order_system_given_equilibrium(  # noqa
            self):
        q_equilibrium = torch.tensor([0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1], dtype=self.dtype)
        dt = 0.01
        forward_system = self.\
            construct_relu_second_order_forward_system_given_equilibrium(
                q_equilibrium, u_equilibrium, dt)

        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network2,
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            torch.tensor([0.1, 0.2, 0.3, 0.4],
                         dtype=self.dtype), np.array([0.]), np.array([1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network2,
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            torch.tensor([-0.1, 0.2, 0.3, 0.5], dtype=self.dtype),
            np.array([-10.]), np.array([1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network2,
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            torch.tensor([0.1, 0.0, 0.0, 0.8],
                         dtype=self.dtype), np.array([0.]), np.array([10.]))

        # Test with controller network3, a simple linear network.
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network3,
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            torch.tensor([0.1, 0.0, 0.2, 0.5], dtype=self.dtype),
            np.array([-10.]), np.array([10.]))

    def test_add_dynamics_mip_constraint_relu_second_order_residue_system_given_equilibrium(  # noqa
            self):
        forward_system = self.\
            construct_relu_second_order_residue_system_given_equilibrium()

        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network2,
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            torch.tensor([0.1, 0.2, 0.3, 0.4],
                         dtype=self.dtype), np.array([0.]), np.array([1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network2,
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            torch.tensor([-0.1, 0.2, 0.3, 0.5], dtype=self.dtype),
            np.array([-10.]), np.array([1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network2,
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            torch.tensor([0.1, 0.0, 0.0, 0.8],
                         dtype=self.dtype), np.array([0.]), np.array([10.]))

    def test_add_dynamics_mip_constraint_relu_system_given_equilibrium(self):
        # Construct a ReLUSystemGivenEquilibrium as the forward dynamical
        # system.
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        forward_system = self.construct_relu_forward_system_given_equilibrium(
            x_equilibrium, u_equilibrium)

        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network1,
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            torch.tensor([0.1, 0.2, 0.3], dtype=self.dtype), np.array([0.,
                                                                       0.]),
            np.array([1., 1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network1,
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            torch.tensor([-0.1, 0.2, 0.3], dtype=self.dtype),
            np.array([-10, -10.]), np.array([1., 1.]))
        self.add_dynamics_mip_constraint_tester(
            forward_system, self.controller_network1,
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            torch.tensor([0.1, 0.0, 0.0], dtype=self.dtype), np.array([0.,
                                                                       0.]),
            np.array([10., 10.]))

    def eval_u(self, dut, x_val):
        """
        Compute u for a single state.
        """
        xhat = compute_xhat._get_xhat_val(x_val, dut.x_equilibrium,
                                          dut.xhat_indices)
        u_pre_sat = dut.controller_network(x_val) - dut.controller_network(
            xhat) + dut.u_equilibrium
        u = u_pre_sat.clone()
        for i in range(u_pre_sat.shape[0]):
            if u_pre_sat[i] < dut.u_lower_limit[i]:
                u[i] = dut.u_lower_limit[i]
            if u_pre_sat[i] > dut.u_upper_limit[i]:
                u[i] = dut.u_upper_limit[i]
        return u

    def test_compute_u(self):
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        forward_system = self.construct_relu_forward_system_given_equilibrium(
            x_equilibrium, u_equilibrium)
        u_lower_limit = np.array([0., -10.])
        u_upper_limit = np.array([100., 0.3])
        closed_loop_system = feedback_system.FeedbackSystem(
            forward_system, self.controller_network1, x_equilibrium,
            u_equilibrium, u_lower_limit, u_upper_limit)

        x_all = (torch.tensor([0.2, 0.4, 0.9], dtype=self.dtype),
                 torch.tensor([-0.5, 1.2, 1.4], dtype=self.dtype),
                 torch.tensor([1.6, -0.4, 2.3], dtype=self.dtype))

        with torch.no_grad():
            for x in x_all:
                u = closed_loop_system.compute_u(x)
                u_expected = self.eval_u(closed_loop_system, x)
                np.testing.assert_allclose(u.detach().numpy(),
                                           u_expected.detach().numpy())

        # Test a batch of x.
        x = torch.tensor([[0.2, 0.4, 0.9], [0.5, 0.1, 1.2]], dtype=self.dtype)
        u = closed_loop_system.compute_u(x)
        self.assertEqual(u.shape, (2, 2))
        for i in range(2):
            np.testing.assert_allclose(
                u[i].detach().numpy(),
                closed_loop_system.compute_u(x[i]).detach().numpy())

        x = torch.tensor([[0.2, 1.4, 0.5], [1.6, -0.1, 1.1], [-2.2, 0.5, 1.4]],
                         dtype=self.dtype)
        u = closed_loop_system.compute_u(x)
        self.assertEqual(u.shape, (3, 2))
        for i in range(3):
            np.testing.assert_allclose(
                u[i].detach().numpy(),
                closed_loop_system.compute_u(x[i]).detach().numpy())

        # set xhat_indices
        closed_loop_system.xhat_indices = [0, 2]
        with torch.no_grad():
            for x in x_all:
                u = closed_loop_system.compute_u(x)
                u_expected = self.eval_u(closed_loop_system, x)
                np.testing.assert_allclose(u.detach().numpy(),
                                           u_expected.detach().numpy())

        # Test a batch of x.
        x = torch.tensor([[0.2, 0.4, 0.9], [0.5, 0.1, 1.2]], dtype=self.dtype)
        u = closed_loop_system.compute_u(x)
        self.assertEqual(u.shape, (2, 2))
        for i in range(2):
            np.testing.assert_allclose(
                u[i].detach().numpy(),
                closed_loop_system.compute_u(x[i]).detach().numpy())

    def step_forward_at_equilibrium_test(self, forward_system,
                                         controller_network, x_equilibrium,
                                         u_equilibrium, u_lower_limit,
                                         u_upper_limit):
        closed_loop_system = feedback_system.FeedbackSystem(
            forward_system, controller_network, x_equilibrium, u_equilibrium,
            u_lower_limit, u_upper_limit)
        with torch.no_grad():
            x_next = closed_loop_system.step_forward(x_equilibrium)
            np.testing.assert_allclose(x_next.detach().numpy(),
                                       x_equilibrium.detach().numpy())

    def step_forward_test(self, forward_system, controller_network, x,
                          x_equilibrium, u_equilibrium, u_lower_limit,
                          u_upper_limit):
        closed_loop_system = feedback_system.FeedbackSystem(
            forward_system, controller_network, x_equilibrium, u_equilibrium,
            u_lower_limit, u_upper_limit)

        with torch.no_grad():
            if len(x.shape) == 1:
                x_next = closed_loop_system.step_forward(x)
                self.assertIsInstance(x_next, torch.Tensor)
                self.assertEqual(x_next.shape, (closed_loop_system.x_dim, ))
                u = closed_loop_system.compute_u(x)
                x_next_expected = forward_system.step_forward(x, u)
                if isinstance(forward_system,
                              hybrid_linear_system.HybridLinearSystem):
                    x_next_expected = x_next_expected[0]
                np.testing.assert_allclose(x_next.detach().numpy(),
                                           x_next_expected.detach().numpy())

            else:
                x_next = closed_loop_system.step_forward(x)
                self.assertIsInstance(x_next, torch.Tensor)
                self.assertEqual(x.shape, x_next.shape)
                for i in range(x.shape[0]):
                    np.testing.assert_allclose(
                        x_next[i].detach().numpy(),
                        closed_loop_system.step_forward(x[i]).detach().numpy())

    def test_step_forward_hybrid_linear_system(self):
        forward_system = self.construct_hybrid_linear_system_example()
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        x = torch.tensor([-1, 0.5, 0.2], dtype=self.dtype)
        self.step_forward_test(forward_system, self.controller_network1, x,
                               x_equilibrium, u_equilibrium,
                               np.array([-np.inf, -np.inf]),
                               np.array([np.inf, np.inf]))

    def test_step_forward_relu_system(self):
        forward_system = self.construct_relu_forward_system()
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        x = torch.tensor([-1, 0.5, 0.2], dtype=self.dtype)
        self.step_forward_test(forward_system, self.controller_network1, x,
                               x_equilibrium, u_equilibrium,
                               np.array([-np.inf, -np.inf]),
                               np.array([np.inf, np.inf]))
        self.step_forward_test(forward_system, self.controller_network1,
                               x, x_equilibrium, u_equilibrium,
                               np.array([-1., 0]), np.array([10., 10.]))
        self.step_forward_test(
            forward_system, self.controller_network1,
            torch.tensor([[0.1, -.5, -0.2], [0.4, -0.3, 0.5]],
                         dtype=self.dtype), x_equilibrium, u_equilibrium,
            np.array([-1, -2.]), np.array([2., 3.]))

    def test_step_forward_relu_system_given_equilibrium(self):
        x_equilibrium = torch.tensor([0, 0.5, 0.3], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1, 0.2], dtype=self.dtype)
        forward_system = self.construct_relu_forward_system_given_equilibrium(
            x_equilibrium, u_equilibrium)
        self.step_forward_at_equilibrium_test(forward_system,
                                              self.controller_network1,
                                              x_equilibrium, u_equilibrium,
                                              np.array([-np.inf, -np.inf]),
                                              np.array([np.inf, np.inf]))
        self.step_forward_at_equilibrium_test(forward_system,
                                              self.controller_network1,
                                              x_equilibrium, u_equilibrium,
                                              np.array([0., -10.]),
                                              np.array([10., 0.3]))
        x = torch.tensor([-1, 0.5, 0.2], dtype=self.dtype)
        self.step_forward_test(forward_system, self.controller_network1, x,
                               forward_system.x_equilibrium,
                               forward_system.u_equilibrium,
                               np.array([-np.inf, -np.inf]),
                               np.array([np.inf, np.inf]))
        self.step_forward_test(forward_system, self.controller_network1, x,
                               forward_system.x_equilibrium,
                               forward_system.u_equilibrium,
                               np.array([-10., -10.]), np.array([0.2, 0.3]))
        self.step_forward_test(
            forward_system, self.controller_network1,
            torch.tensor([[0.1, -0.4, 0.5], [0.2, -0.5, 0.3]],
                         dtype=self.dtype), forward_system.x_equilibrium,
            forward_system.u_equilibrium, np.array([-10., -10.]),
            np.array([0.2, 0.3]))

    def test_step_forward_relu_second_order_system_given_equilibrium(self):
        q_equilibrium = torch.tensor([0.5, 0.4], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1], dtype=self.dtype)
        dt = 0.01

        forward_system = self.\
            construct_relu_second_order_forward_system_given_equilibrium(
                q_equilibrium, u_equilibrium, dt)
        self.step_forward_at_equilibrium_test(forward_system,
                                              self.controller_network2,
                                              forward_system.x_equilibrium,
                                              u_equilibrium,
                                              np.array([-np.inf]),
                                              np.array([np.inf]))
        self.step_forward_at_equilibrium_test(forward_system,
                                              self.controller_network2,
                                              forward_system.x_equilibrium,
                                              u_equilibrium, np.array([0.]),
                                              np.array([10.]))
        self.step_forward_test(
            forward_system, self.controller_network2,
            torch.tensor([-0.3, 0.4, 0.2, 1.5], dtype=self.dtype),
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            np.array([-np.inf]), np.array([np.inf]))
        self.step_forward_test(
            forward_system, self.controller_network2,
            torch.tensor([-0.3, 0.7, 0.25, 1.1], dtype=self.dtype),
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            np.array([-10.]), np.array([0.2]))
        self.step_forward_test(
            forward_system, self.controller_network2,
            torch.tensor([-0.8, 1.4, 0.4, 1.2], dtype=self.dtype),
            torch.tensor([0.2, -0.1, 0.5, 0.3], dtype=self.dtype),
            torch.tensor([0.2], dtype=self.dtype), np.array([-10.]),
            np.array([0.2]))
        self.step_forward_test(
            forward_system, self.controller_network2,
            torch.tensor([[-.3, .7, 0.25, 1.1], [.3, .2, -.1, -0.5]],
                         dtype=self.dtype), forward_system.x_equilibrium,
            forward_system.u_equilibrium, np.array([-10.]), np.array([0.2]))

    def test_step_forward_relu_second_order_residue_system_given_equilibrium(
            self):  # noqa
        forward_system = self.\
            construct_relu_second_order_residue_system_given_equilibrium()
        self.step_forward_at_equilibrium_test(forward_system,
                                              self.controller_network2,
                                              forward_system.x_equilibrium,
                                              forward_system.u_equilibrium,
                                              np.array([-np.inf]),
                                              np.array([np.inf]))
        self.step_forward_at_equilibrium_test(forward_system,
                                              self.controller_network2,
                                              forward_system.x_equilibrium,
                                              forward_system.u_equilibrium,
                                              np.array([0.]), np.array([10.]))
        self.step_forward_test(
            forward_system, self.controller_network2,
            torch.tensor([-0.3, 0.4, 0.2, 1.5], dtype=self.dtype),
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            np.array([-np.inf]), np.array([np.inf]))
        self.step_forward_test(
            forward_system, self.controller_network2,
            torch.tensor([-0.3, 0.7, 0.25, 1.1], dtype=self.dtype),
            forward_system.x_equilibrium, forward_system.u_equilibrium,
            np.array([-10.]), np.array([0.2]))
        self.step_forward_test(
            forward_system, self.controller_network2,
            torch.tensor([-0.8, 1.4, 0.4, 1.2], dtype=self.dtype),
            torch.tensor([0.2, -0.1, 0.5, 0.3], dtype=self.dtype),
            torch.tensor([0.2], dtype=self.dtype), np.array([-10.]),
            np.array([0.2]))
        self.step_forward_test(
            forward_system, self.controller_network2,
            torch.tensor([[-.3, .7, 0.25, 1.1], [.3, .2, -.1, -0.5]],
                         dtype=self.dtype), forward_system.x_equilibrium,
            forward_system.u_equilibrium, np.array([-10.]), np.array([0.2]))

    def add_controller_mip_constraint_tester(self, dut, x_val):
        mip = gurobi_torch_mip.GurobiTorchMILP(torch.float64)
        x_var = mip.addVars(dut.x_dim,
                            lb=-gurobipy.GRB.INFINITY,
                            vtype=gurobipy.GRB.CONTINUOUS,
                            name="x")
        u_var = mip.addVars(dut.forward_system.u_dim,
                            lb=-gurobipy.GRB.INFINITY,
                            vtype=gurobipy.GRB.CONTINUOUS,
                            name="u")
        controller_mip_cnstr_return = dut._add_controller_mip_constraint(
            mip, x_var, u_var, "slack", "binary", lp_relaxation=False)
        for v in controller_mip_cnstr_return.binary:
            self.assertEqual(v.vtype, gurobipy.GRB.BINARY)
        # Now add constraint on x_var = x_val
        mip.addMConstrs([torch.eye(dut.x_dim, dtype=torch.float64)], [x_var],
                        b=x_val,
                        sense=gurobipy.GRB.EQUAL)
        mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        mip.gurobi_model.optimize()

        u_val = np.array([u.x for u in u_var])
        u_val_expected = dut.compute_u(x_val)
        np.testing.assert_allclose(u_val, u_val_expected.detach().numpy())

        # Now test with lp_relaxation=True
        lp = gurobi_torch_mip.GurobiTorchMILP(torch.float64)
        x_var_lp = lp.addVars(dut.x_dim,
                              lb=-gurobipy.GRB.INFINITY,
                              vtype=gurobipy.GRB.CONTINUOUS,
                              name="x")
        u_var_lp = lp.addVars(dut.forward_system.u_dim,
                              lb=-gurobipy.GRB.INFINITY,
                              vtype=gurobipy.GRB.CONTINUOUS,
                              name="u")
        controller_mip_cnstr_return_lp = dut._add_controller_mip_constraint(
            lp, x_var_lp, u_var_lp, "slack", "binary", lp_relaxation=True)
        self.assertEqual(len(lp.zeta), 0)
        for v in controller_mip_cnstr_return_lp.binary:
            self.assertEqual(v.vtype, gurobipy.GRB.CONTINUOUS)
            self.assertEqual(v.lb, 0.)
            self.assertEqual(v.ub, 1.)
        np.testing.assert_allclose(
            controller_mip_cnstr_return.u_lower_bound.detach().numpy(),
            controller_mip_cnstr_return_lp.u_lower_bound.detach().numpy())
        np.testing.assert_allclose(
            controller_mip_cnstr_return.u_upper_bound.detach().numpy(),
            controller_mip_cnstr_return_lp.u_upper_bound.detach().numpy())

    def test_add_controller_mip_constraint(self):
        """
        Test _add_controller_mip_constraint when the controller_network is
        1. a torch.nn.Sequential, with both linear and relu layers.
        2. a torch.nn.Linear layer.
        """
        q_equilibrium = torch.tensor([0.5, 0.4], dtype=self.dtype)
        u_equilibrium = torch.tensor([0.1], dtype=self.dtype)
        dt = 0.01

        forward_system = self.\
            construct_relu_second_order_forward_system_given_equilibrium(
                q_equilibrium, u_equilibrium, dt)
        for controller_network in (self.controller_network2,
                                   self.controller_network3):
            u_bounds = [(np.array([-20.]), np.array([20.]))]
            u_bounds.append((np.array([1.]), np.array([10.])))
            u_bounds.append((np.array([-np.inf]), np.array([np.inf])))
            for (u_lo, u_up) in u_bounds:
                dut = feedback_system.FeedbackSystem(
                    forward_system, controller_network,
                    forward_system.x_equilibrium,
                    torch.tensor([2.], dtype=self.dtype), u_lo, u_up)
                self.add_controller_mip_constraint_tester(
                    dut, torch.tensor([2., 0.5, 0.6, 3.1], dtype=self.dtype))
                self.add_controller_mip_constraint_tester(
                    dut, torch.tensor([0.4, -1.5, 3.6, 3.1], dtype=self.dtype))
                if not isinstance(controller_network, torch.nn.Linear):
                    dut.xhat_indices = [0, 2]
                    self.add_controller_mip_constraint_tester(
                        dut, torch.tensor([2., 0.5, 0.6, 3.1],
                                          dtype=self.dtype))
                    self.add_controller_mip_constraint_tester(
                        dut,
                        torch.tensor([0.4, -1.5, 3.6, 3.1], dtype=self.dtype))


class TestAddInputSaturationConstraint(unittest.TestCase):
    def add_input_saturation_constraint_tester(self, u_lower_limit,
                                               u_upper_limit, u_pre_sat_lo,
                                               u_pre_sat_up):
        dtype = torch.float64
        mip = gurobi_torch_mip.GurobiTorchMIP(dtype)
        u_dim = len(u_lower_limit)
        u_var = mip.addVars(u_dim, lb=-gurobipy.GRB.INFINITY)
        u_pre_sat = mip.addVars(u_dim, lb=-gurobipy.GRB.INFINITY)
        u_lower_bound, u_upper_bound = \
            feedback_system._add_input_saturation_constraint(
                mip, u_var, u_pre_sat, u_lower_limit, u_upper_limit,
                u_pre_sat_lo, u_pre_sat_up, dtype, lp_relaxation=False)
        for i in range(u_dim):
            self.assertEqual(
                u_lower_bound[i].item(),
                torch.clamp(u_pre_sat_lo[i], u_lower_limit[i],
                            u_upper_limit[i]).item())
            self.assertEqual(
                u_upper_bound[i].item(),
                torch.clamp(u_pre_sat_up[i], u_lower_limit[i],
                            u_upper_limit[i]).item())
        # Now take many samples of u_pre, make sure u_var is the result of the
        # saturation.
        u_pre_sat_samples = utils.uniform_sample_in_box(
            u_pre_sat_lo, u_pre_sat_up, 100)
        for i in range(u_pre_sat_samples.shape[0]):
            for j in range(u_dim):
                u_pre_sat[j].lb = u_pre_sat_samples[i, j]
                u_pre_sat[j].ub = u_pre_sat_samples[i, j]
            mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            mip.gurobi_model.optimize()
            self.assertEqual(mip.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            u_val = u_pre_sat_samples[i]
            for j in range(u_dim):
                if u_val[j] > u_upper_limit[j]:
                    u_val[j] = u_upper_limit[j]
                elif u_val[j] < u_lower_limit[j]:
                    u_val[j] = u_lower_limit[j]
            np.testing.assert_allclose(np.array([v.x for v in u_var]), u_val)

        # Now test lp_relaxation=True
        lp = gurobi_torch_mip.GurobiTorchMIP(dtype)
        u_var_lp = lp.addVars(u_dim, lb=-gurobipy.GRB.INFINITY)
        u_pre_sat_lp = lp.addVars(u_dim, lb=-gurobipy.GRB.INFINITY)
        u_lower_bound_lp, u_upper_bound_lp = \
            feedback_system._add_input_saturation_constraint(
                lp, u_var_lp, u_pre_sat_lp, u_lower_limit, u_upper_limit,
                u_pre_sat_lo, u_pre_sat_up, dtype, lp_relaxation=True)
        self.assertEqual(len(lp.zeta), 0)
        np.testing.assert_allclose(u_lower_bound.detach().numpy(),
                                   u_lower_bound_lp.detach().numpy())
        np.testing.assert_allclose(u_upper_bound.detach().numpy(),
                                   u_upper_bound_lp.detach().numpy())

    def test_add_input_saturation_constraints(self):
        dtype = torch.float64
        self.add_input_saturation_constraint_tester(
            np.array([-np.inf, 1.]), np.array([np.inf, 3.]),
            torch.tensor([0.5, 0.2], dtype=dtype),
            torch.tensor([0.9, 5.1], dtype=dtype))
        self.add_input_saturation_constraint_tester(
            np.array([1, -10, -2.]), np.array([5., 2., 5.]),
            torch.tensor([0.5, 2.1, 4.5], dtype=dtype),
            torch.tensor([0.9, 4., 5.2], dtype=dtype))


if __name__ == "__main__":
    unittest.main()
