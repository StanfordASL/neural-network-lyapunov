import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.adversarial_sample as adversarial_sample
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.model_bounds as model_bounds
import double_integrator

import numpy as np
import unittest
import torch
import torch.nn as nn


class AdversarialSampleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        np.random.seed(123)
        dt = 1.
        dtype = torch.float64
        self.dtype = dtype
        (A_c, B_c) = double_integrator.double_integrator_dynamics(dtype)
        x_dim = A_c.shape[1]
        self.x_dim = x_dim
        u_dim = B_c.shape[1]
        A = torch.eye(x_dim, dtype=dtype) + dt * A_c
        B = dt * B_c
        sys = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)
        c = torch.zeros(x_dim, dtype=dtype)
        P = torch.cat((-torch.eye(x_dim+u_dim),
                       torch.eye(x_dim+u_dim)), 0).type(dtype)
        x_lo = -10. * torch.ones(x_dim, dtype=dtype)
        x_up = 10. * torch.ones(x_dim, dtype=dtype)
        u_lo = -1. * torch.ones(u_dim, dtype=dtype)
        u_up = 1. * torch.ones(u_dim, dtype=dtype)
        q = torch.cat((-x_lo, -u_lo, x_up, u_up), 0).type(dtype)
        sys.add_mode(A, B, c, P, q)
        R = torch.eye(sys.u_dim)
        Q = torch.eye(sys.x_dim)
        N = 5
        vf = value_to_optimization.ValueFunction(sys, N,
                                                 x_lo, x_up, u_lo, u_up)
        vf.set_cost(Q=Q, R=R)
        vf.set_terminal_cost(Qt=Q, Rt=R)
        self.vf = vf
        self.V = vf.get_value_function()

        linear1 = nn.Linear(x_dim, 10)
        linear1.weight.data = torch.tensor(
            np.random.rand(10, x_dim), dtype=dtype)
        linear1.bias.data = torch.tensor(
            np.random.rand(10), dtype=dtype)
        linear2 = nn.Linear(10, 10)
        linear2.weight.data = torch.tensor(
            np.random.rand(10, 10),
            dtype=dtype)
        linear2.bias.data = torch.tensor(
            np.random.rand(10), dtype=dtype)
        linear3 = nn.Linear(10, 1)
        linear3.weight.data = torch.tensor(
            np.random.rand(1, 10), dtype=dtype)
        linear3.bias.data = torch.tensor([-10], dtype=dtype)
        self.model = nn.Sequential(linear1, nn.ReLU(), linear2,
                                   nn.ReLU(), linear3)
        self.x0_lo = -1. * torch.ones(x_dim, dtype=dtype)
        self.x0_up = 1. * torch.ones(x_dim, dtype=dtype)
        self.as_generator = adversarial_sample.AdversarialSampleGenerator(
            vf, self.x0_lo, self.x0_up)

    def test_upper_bound_global(self):
        for requires_grad in [True, False]:
            (eps_adv, x_adv) = self.as_generator.get_upper_bound_global(
                self.model, requires_grad=requires_grad)
            eps_expected = self.V(x_adv)[0] - self.model(x_adv)
            self.assertAlmostEqual(eps_adv.item(),
                                   eps_expected.item(), places=5)
            self.assertTrue(torch.all(x_adv <= self.x0_up))
            self.assertTrue(torch.all(x_adv >= self.x0_lo))
        with torch.no_grad():
            for i in range(20):
                x0 = torch.rand(self.x_dim, dtype=self.dtype) *\
                    (self.x0_up - self.x0_lo) + self.x0_lo
                eps_sample = self.V(x0)[0] - self.model(x0)
                self.assertLessEqual(eps_adv.item(), eps_sample.item())

    def test_setup_val_opt(self):
        for i in range(10):
            x0 = torch.rand(self.x_dim, dtype=self.dtype) * \
                (self.x0_up - self.x0_lo) + self.x0_lo
            v_exp = self.V(x0)[0]
            (prob, x, s, alpha) = self.as_generator.setup_val_opt(x_val=x0)
            prob.gurobi_model.optimize()
            v = prob.gurobi_model.objVal
            self.assertAlmostEqual(v, v_exp, places=5)

    def test_upper_bound_sample(self):
        (eps_glob, x_glob) = self.as_generator.get_upper_bound_global(
            self.model, requires_grad=False)
        (eps_adv, x_adv, V_adv) = self.as_generator.get_upper_bound_sample(
            self.model, num_iter=15, learning_rate=.1)
        self.assertAlmostEqual(eps_adv[-1, 0].item(), eps_glob.item(),
                               places=5)

    def test_lower_bound_sample(self):
        (eps_adv, x_adv, V_adv) = self.as_generator.get_lower_bound_sample(
            self.model, num_iter=15, learning_rate=.1)
        with torch.no_grad():
            for i in range(20):
                # note that the property is actually only guaranteed locally!
                x0 = torch.rand(self.x_dim, dtype=self.dtype) *\
                    (self.x0_up - self.x0_lo) + self.x0_lo
                eps_sample = self.V(x0)[0] - self.model(x0)
                self.assertGreaterEqual(eps_adv[-1, 0].item(),
                                        eps_sample.item())

    def test_v_with_grad(self):
        x_adv0 = torch.rand(self.x_dim, dtype=self.dtype) *\
            (self.x0_up - self.x0_lo) + self.x0_lo
        (prob, x, s, alpha) = self.as_generator.setup_val_opt(x_val=x_adv0)
        prob.gurobi_model.optimize()
        Vx_expect = prob.compute_objective_from_mip_data_and_solution(
            penalty=1e-6)
        Vx = self.as_generator.V_with_grad(x_adv0)
        self.assertAlmostEqual(Vx.item(), Vx_expect.item(), places=4)

    def test_squared_bound_sample(self):
        x_adv0 = torch.rand(self.x_dim, dtype=self.dtype) *\
            (self.x0_up - self.x0_lo) + self.x0_lo
        max_iter = 20
        (eps_adv, x_adv,
         V_adv, _) = self.as_generator.get_squared_bound_sample(
            self.model, max_iter=max_iter, conv_tol=1e-4, learning_rate=.1,
            x_adv0=x_adv0)
        self.assertLess(eps_adv.shape[0], max_iter)
        with torch.no_grad():
            for i in range(20):
                # note that the property is actually only guaranteed locally!
                x0 = torch.rand(self.x_dim, dtype=self.dtype) *\
                    (self.x0_up - self.x0_lo) + self.x0_lo
                eps_sample = torch.pow(self.V(x0)[0] - self.model(x0), 2)
                self.assertGreaterEqual(eps_adv[-1, 0].item(),
                                        eps_sample.item())

    def test_setup_eps_opt(self):
        mb = model_bounds.ModelBounds(self.vf, self.model)
        eps_opt_coeffs = mb.epsilon_opt(self.model, self.x0_lo, self.x0_up)
        (prob, x, y, gamma) = self.as_generator.setup_eps_opt(eps_opt_coeffs)
        prob.gurobi_model.optimize()
        epsilon1 = prob.compute_objective_from_mip_data_and_solution(
            penalty=1e-8)
        x_val = torch.Tensor([k.x for k in x]).type(self.dtype)
        (prob, x, y, gamma) = self.as_generator.setup_eps_opt(eps_opt_coeffs,
                                                              x_val=x_val)
        prob.gurobi_model.optimize()
        epsilon2 = prob.compute_objective_from_mip_data_and_solution(
            penalty=1e-8)
        self.assertAlmostEqual(epsilon1.item(), epsilon2.item(), places=5)


if __name__ == '__main__':
    unittest.main()
