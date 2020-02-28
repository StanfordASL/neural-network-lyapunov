import robust_value_approx.samples_generator as samples_generator
import robust_value_approx.value_approximation as value_approximation
import double_integrator_utils
import acrobot_utils
import pendulum_utils

import unittest
import torch


class RandomSampleTest(unittest.TestCase):
    def test_random_sample(self):
        N = 5
        vf = double_integrator_utils.get_value_function(N=N)
        x0_lo = -1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        x0_up = 1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        rs_gen = samples_generator.RandomSampleGenerator(vf, x0_lo, x0_up)
        (rand_data, rand_label) = rs_gen.generate_samples(10)
        for n in range(N-1):
            vf_ = double_integrator_utils.get_value_function(N=N-n)
            V_ = vf_.get_value_function()
            for k in range(rand_data.shape[0]):
                x0 = rand_data[k, vf.sys.x_dim*n:vf.sys.x_dim*(n+1)]
                v_ = V_(x0)[0]
                self.assertAlmostEqual(rand_label[k, n].item(), v_, places=5)


class MIPAdvSampleTest(unittest.TestCase):
    def test_mip_grad(self):
        N = 10
        vf = double_integrator_utils.get_value_function(N=N)
        Q = torch.rand((vf.sys.x_dim, vf.sys.x_dim), dtype=vf.dtype)
        R = torch.rand((vf.sys.u_dim, vf.sys.u_dim), dtype=vf.dtype)
        Q = Q.t()@Q + torch.eye(vf.sys.x_dim, dtype=vf.dtype)
        R = R.t()@R + torch.eye(vf.sys.u_dim, dtype=vf.dtype)
        q = torch.rand(vf.sys.x_dim, dtype=vf.dtype)
        r = torch.rand(vf.sys.u_dim, dtype=vf.dtype)
        vf.set_cost(Q=Q, R=R, q=q*0, r=r*0)
        vf.set_terminal_cost(Qt=Q, Rt=R, qt=q*0, rt=r*0)
        x0_lo = -1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        x0_up = 1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        as_gen = samples_generator.AdversarialSampleGenerator(
            vf, x0_lo, x0_up)
        eps = 1e-2
        for k in range(10):
            x0 = torch.rand(vf.sys.x_dim, dtype=vf.dtype) *\
                (x0_up - x0_lo) + x0_lo
            x0.requires_grad = True
            v1, v2 = as_gen.V_with_grad(x0)
            obj_grad = torch.autograd.grad(v2[0], x0)[0]
            for i in range(vf.sys.x_dim):
                x0_ = x0.clone()
                x0_[i] += eps
                v1_, v2_ = as_gen.V_with_grad(x0_)
                obj_grad_1 = (v2_[0] - v2[0]) / eps
                x0_ = x0.clone()
                x0_[i] -= eps
                v1_, v2_ = as_gen.V_with_grad(x0_)
                obj_grad_2 = (v2_[0] - v2[0]) / -eps
                obj_grad_ = .5 * (obj_grad_1 + obj_grad_2)
                self.assertAlmostEqual(obj_grad_.item(), obj_grad[i].item(),
                                       places=4)

    def test_squared_bound_sample(self):
        N = 5
        vf = double_integrator_utils.get_value_function(N=N)
        V = vf.get_value_function()
        x0_lo = -1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        x0_up = 1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        max_iter = 100
        as_gen = samples_generator.AdversarialSampleGenerator(
            vf, x0_lo, x0_up, max_iter=max_iter, learning_rate=.1)
        value_approx =\
            value_approximation.FiniteHorizonValueFunctionApproximation(
                vf, x0_lo, x0_up, 16, 1)
        x_adv0 = torch.Tensor([.1, .1]).type(vf.dtype)
        (epsilon_buff,
         x_adv_buff,
         cost_to_go_buff) = as_gen.get_squared_bound_sample(
            value_approx, x_adv0)
        self.assertLess(epsilon_buff.shape[0], max_iter)
        with torch.no_grad():
            for i in range(20):
                # note that the property is actually only guaranteed locally!
                x0 = torch.rand(vf.sys.x_dim, dtype=vf.dtype) *\
                    (x0_up - x0_lo) + x0_lo
                eps_sample = torch.pow(V(x0)[0] - value_approx.eval(
                    0, x0.unsqueeze(0)), 2)
                self.assertGreaterEqual(epsilon_buff[-1, 0].item(),
                                        eps_sample.item())

    def test_generate_samples(self):
        N = 5
        vf = double_integrator_utils.get_value_function(N=N)
        x0_lo = -1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        x0_up = 1 * torch.ones(vf.sys.x_dim, dtype=vf.dtype)
        as_gen = samples_generator.AdversarialSampleGenerator(
            vf, x0_lo, x0_up, max_iter=5)
        value_approx =\
            value_approximation.FiniteHorizonValueFunctionApproximation(
                vf, x0_lo, x0_up, 16, 1)
        n = 10
        (adv_data, adv_label) = as_gen.generate_samples(n, value_approx)
        self.assertEqual(adv_data.shape[0], n)
        self.assertEqual(adv_label.shape[0], n)


class NLPAdvSampleTest(unittest.TestCase):
    def test_nlp_grad(self):
        N = 5
        vf = acrobot_utils.get_value_function(N)
        # vf = pendulum_utils.get_value_function(N)
        x0_lo = -1 * torch.ones(vf.x_dim[0], dtype=vf.dtype)
        x0_up = 1 * torch.ones(vf.x_dim[0], dtype=vf.dtype)
        V_with_grad = vf.get_differentiable_value_function()
        eps = 1e-3
        for k in range(1):
            x0 = torch.rand(vf.x_dim[0], dtype=vf.dtype) *\
                (x0_up - x0_lo) + x0_lo
            x0.requires_grad = True
            v1, v2 = V_with_grad(x0)
            obj_grad = torch.autograd.grad(v2[0], x0)[0]
            for i in range(vf.sys.x_dim):
                x0_ = x0.clone()
                x0_[i] += eps
                v1_, v2_ = V_with_grad(x0_)
                obj_grad_1 = (v2_[0] - v2[0]) / eps
                x0_ = x0.clone()
                x0_[i] -= eps
                v1_, v2_ = V_with_grad(x0_)
                obj_grad_2 = (v2_[0] - v2[0]) / -eps
                obj_grad_ = .5 * (obj_grad_1 + obj_grad_2)
                # self.assertAlmostEqual(obj_grad_.item(), obj_grad[i].item(),
                                       # places=1)
                print(obj_grad_.item())
                print(obj_grad[i].item())
                print("====")

    def test_squared_bound_sample(self):
        N = 5
        vf = acrobot_utils.get_value_function(N)
        V = vf.get_value_function()
        x0_lo = -1 * torch.ones(vf.x_dim[0], dtype=vf.dtype)
        x0_up = 1 * torch.ones(vf.x_dim[0], dtype=vf.dtype)
        max_iter = 100
        as_gen = samples_generator.AdversarialSampleGenerator(
            vf, x0_lo, x0_up, max_iter=max_iter, learning_rate=.1)
        value_approx =\
            value_approximation.FiniteHorizonValueFunctionApproximation(
                vf, x0_lo, x0_up, 16, 1)
        x_adv0 = torch.Tensor([.1, .1, .1, .1]).type(vf.dtype)
        (epsilon_buff,
         x_adv_buff,
         cost_to_go_buff) = as_gen.get_squared_bound_sample(
            value_approx, x_adv0)
        self.assertLess(epsilon_buff.shape[0], max_iter)
        with torch.no_grad():
            for i in range(20):
                # note that the property is actually only guaranteed locally!
                x0 = torch.rand(vf.x_dim[0], dtype=vf.dtype) *\
                    (x0_up - x0_lo) + x0_lo
                eps_sample = torch.pow(V(x0)[0] - value_approx.eval(
                    0, x0.unsqueeze(0)), 2)
                self.assertGreaterEqual(epsilon_buff[-1, 0].item(),
                                        eps_sample.item())

    def test_generate_samples(self):
        N = 5
        vf = acrobot_utils.get_value_function(N)
        x0_lo = -1 * torch.ones(vf.x_dim[0], dtype=vf.dtype)
        x0_up = 1 * torch.ones(vf.x_dim[0], dtype=vf.dtype)
        as_gen = samples_generator.AdversarialSampleGenerator(
            vf, x0_lo, x0_up, max_iter=5)
        value_approx =\
            value_approximation.FiniteHorizonValueFunctionApproximation(
                vf, x0_lo, x0_up, 16, 1)
        n = 10
        (adv_data, adv_label) = as_gen.generate_samples(n, value_approx)
        self.assertEqual(adv_data.shape[0], n)
        self.assertEqual(adv_label.shape[0], n)


if __name__ == '__main__':
    unittest.main()
