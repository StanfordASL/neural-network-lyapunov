import neural_network_lyapunov.examples.car.rrt_star as rrt_star
import neural_network_lyapunov.examples.car.unicycle as unicycle
import numpy as np
import torch
import unittest


class TestRrtStar(unittest.TestCase):
    def setUp(self):
        self.plant = unicycle.Unicycle(torch.float64)
        self.u_lo = np.array([-3, -0.25 * np.pi])
        self.u_up = np.array([6, 0.25 * np.pi])

    def test_constructor(self):
        x_goal = np.array([0, 0., 0.])
        dut = rrt_star.RrtStar(self.plant, 3, self.u_lo, self.u_up, x_goal)
        np.testing.assert_allclose(dut.node_state, x_goal.reshape((1, -1)))
        self.assertListEqual(dut.node_parent, [None])
        self.assertListEqual(dut.node_children, [set()])
        np.testing.assert_allclose(dut.node_cost_to_root, np.array([0.]))
        np.testing.assert_allclose(dut.node_cost_to_parent, np.array([0.]))

    def test_add_node(self):
        x_goal = np.array([0, 0., 0.])
        dut = rrt_star.RrtStar(self.plant, 3, self.u_lo, self.u_up, x_goal)
        x_node1 = np.array([3, 2, 1])
        node1_idx = dut._add_node(x_node1, parent_idx=0, cost_to_parent=4.)
        self.assertEqual(node1_idx, 1)
        np.testing.assert_allclose(dut.node_state, np.vstack(
            (x_goal, x_node1)))
        self.assertEqual(dut.node_state.shape, (2, 3))
        self.assertListEqual(dut.node_parent, [None, 0])
        self.assertListEqual(dut.node_children, [{1}, set()])
        np.testing.assert_allclose(dut.node_cost_to_root, np.array([0., 4.]))
        np.testing.assert_allclose(dut.node_cost_to_parent, np.array([0., 4.]))
        x_node2 = np.array([4, 5., 6.])
        node2_idx = dut._add_node(x_node2, parent_idx=1, cost_to_parent=3.)
        self.assertEqual(node2_idx, 2)
        np.testing.assert_allclose(dut.node_state,
                                   np.vstack((x_goal, x_node1, x_node2)))
        self.assertListEqual(dut.node_parent, [None, 0, 1])
        self.assertListEqual(dut.node_children, [{1}, {2}, set()])
        np.testing.assert_allclose(dut.node_cost_to_root,
                                   np.array([0., 4., 7.]))
        np.testing.assert_allclose(dut.node_cost_to_parent,
                                   np.array([0., 4., 3.]))

    def test_state_distance(self):
        x_goal = np.array([0, 0., 0.])
        dut = rrt_star.RrtStar(self.plant, 3, self.u_lo, self.u_up, x_goal)

        def eval_distance(x1, x2):
            return (x1 - x2) @ (Q @ (x1 - x2))

        Q = np.diag([1., 1., 0.5])
        x1 = np.array([1., 2., 3.])
        x2 = np.array([3., 1., 2.])
        np.testing.assert_allclose(dut.state_distance(x1, x2),
                                   np.array([eval_distance(x1, x2)]))
        x1 = np.array([[1., 2., 3.], [4., -2., -1.]])
        np.testing.assert_allclose(
            dut.state_distance(x1, x2),
            np.array([eval_distance(x1[0], x2),
                      eval_distance(x1[1], x2)]))
        np.testing.assert_allclose(
            dut.state_distance(x2, x1),
            np.array([eval_distance(x1[0], x2),
                      eval_distance(x1[1], x2)]))

    def test_extend_node(self):
        x_goal = np.array([0, 0., 0.])
        dut = rrt_star.RrtStar(self.plant, 3, self.u_lo, self.u_up, x_goal)
        x_node = np.array([1., 2., 3.])
        dut._add_node(x_node, 0, 1.)
        torch.manual_seed(0)
        x_sample = np.array([0.5, 0.3, 0.1])
        x_extend = dut.extend_node(1, 20, 0.05, x_sample)
        self.assertIsInstance(x_extend, np.ndarray)

    def test_find_path(self):
        x_goal = np.array([0, 0., 0.])
        dut = rrt_star.RrtStar(self.plant, 3, self.u_lo, self.u_up, x_goal)

        x1 = np.array([0.5, 0.3, 1.2])
        x2 = np.array([0.5, 0.3, 1.2])
        path_return = dut.find_path(x1, x2)
        self.assertTrue(np.isfinite(path_return[0]))
        np.testing.assert_allclose(path_return[1][:, 0], x1)
        np.testing.assert_allclose(path_return[1][:, -1], x2)
        self.assertAlmostEqual(path_return[0], 0)

        dut._add_node(x1, 0, 0.5)
        x3 = dut.extend_node(1, 100, 0.04, x2)
        path_return = dut.find_path(x1, x3)
        self.assertTrue(np.isfinite(path_return[0]))
        np.testing.assert_allclose(path_return[1][:, 0], x1)
        np.testing.assert_allclose(path_return[1][:, -1], x3)
        for i in range(path_return[1].shape[1] - 1):
            xdot_l = dut.plant.dynamics(path_return[1][:, i],
                                        path_return[2][:, i])
            xdot_r = dut.plant.dynamics(path_return[1][:, i + 1],
                                        path_return[2][:, i + 1])
            np.testing.assert_allclose(
                path_return[1][:, i + 1] - path_return[1][:, i],
                (xdot_l + xdot_r) * path_return[3][i] / 2,
                atol=1E-6)

        x4 = np.array([1.5, 2.1, 3.4])
        path_return = dut.find_path(x1, x4)
        self.assertFalse(np.isfinite(path_return[0]))

    def test_update_parent(self):
        x_goal = np.array([0, 0., 0.])
        dut = rrt_star.RrtStar(self.plant, 3, self.u_lo, self.u_up, x_goal)

        dut._add_node(np.array([1., 2., 3.]), 0, 1.)
        dut._add_node(np.array([1., 2., 4.]), 0, 2.)

        dut.update_parent(node_idx=2,
                          parent_idx=1,
                          cost_to_parent=0.5,
                          path_to_parent_x=None,
                          path_to_parent_u=None,
                          path_to_parent_dt=None)
        self.assertListEqual(dut.node_parent, [None, 0, 1])
        self.assertListEqual(dut.node_children, [{1}, {2}, set()])
        np.testing.assert_allclose(dut.node_cost_to_root,
                                   np.array([0., 1., 1.5]))
        np.testing.assert_allclose(dut.node_cost_to_parent,
                                   np.array([0., 1., 0.5]))

        dut._add_node(np.array([1., 3., 5]), 0, 3)
        dut._add_node(np.array([0.5, 2, 4]), 1, 2.)
        dut._add_node(np.array([2., 3., 1.]), 4, 0.6)
        dut._add_node(np.array([0.1, 3.5, 3.2]), 3, 4.)

        dut.update_parent(node_idx=1,
                          parent_idx=3,
                          cost_to_parent=2.5,
                          path_to_parent_x=None,
                          path_to_parent_u=None,
                          path_to_parent_dt=None)
        self.assertListEqual(dut.node_parent, [None, 3, 1, 0, 1, 4, 3])
        self.assertListEqual(
            dut.node_children,
            [{3}, {2, 4}, set(), {1, 6}, {5},
             set(), set()])
        np.testing.assert_allclose(dut.node_cost_to_root,
                                   np.array([0, 5.5, 6, 3, 7.5, 8.1, 7]))
        np.testing.assert_allclose(dut.node_cost_to_parent,
                                   np.array([0, 2.5, 0.5, 3, 2, 0.6, 4.]))

    def test_connect_state_extend(self):
        x_goal = np.array([0, 0., 0.])
        dut = rrt_star.RrtStar(self.plant, 3, self.u_lo, self.u_up, x_goal)

        node1 = dut._add_node(np.array([1., 2., 3.]), 0, 1.)
        dut._add_node(np.array([1., 2., 4.]), 0, 2.)
        node3_x = dut.extend_node(node1, 20, 0.01, np.array([1.5, 0.4, 0.]))
        node3 = dut._add_node(node3_x, node1, 1E-10)
        node4_x = dut.extend_node(node1, 20, 0.01, np.array([2.5, 2.4, 0.]))
        node4 = dut._add_node(node4_x, node1, 1E5)
        node5 = dut._add_node(np.array([2.5, 0.3, 0.2]), node1, 1.)

        x_extend = dut.extend_node(node1, 20, 0.02, np.array([1.5, 0.1, 0.]))
        neighbour_indices = [node1, node3, node4, node5]
        new_node_idx = dut.connect_state_extend(x_extend, neighbour_indices)
        self.assertIsNotNone(new_node_idx)
        np.testing.assert_allclose(dut.node_state[new_node_idx], x_extend)
        cost_to_node = {}
        cost_via_node = {}
        old_cost_to_root = {}
        old_parent = {}
        for i in neighbour_indices:
            cost_to_node[i] = dut.find_path(x_extend, dut.node_state[i])[0]
            cost_via_node[i] = dut.node_cost_to_root[i] + cost_to_node[i]
            old_cost_to_root[i] = dut.node_cost_to_root[i]
            old_parent[i] = dut.node_parent[i]
        self.assertTrue(np.isfinite(cost_to_node[node3]))
        self.assertTrue(np.isfinite(cost_to_node[node4]))
        for i in neighbour_indices:
            if i != dut.node_parent[new_node_idx]:
                self.assertGreater(
                    cost_via_node[i],
                    cost_via_node[dut.node_parent[new_node_idx]])
            else:
                self.assertAlmostEqual(cost_via_node[i],
                                       dut.node_cost_to_root[new_node_idx])
                self.assertAlmostEqual(cost_to_node[i],
                                       dut.node_cost_to_parent[new_node_idx])
            if cost_to_node[i] + dut.node_cost_to_root[
                    new_node_idx] < old_cost_to_root[i]:
                # update the parent.
                self.assertEqual(dut.node_parent[i], new_node_idx)
                self.assertAlmostEqual(
                    dut.node_cost_to_root[i],
                    cost_to_node[i] + dut.node_cost_to_root[new_node_idx])
            else:
                self.assertEqual(dut.node_parent[i], old_parent[i])
                self.assertAlmostEqual(dut.node_cost_to_root[i],
                                       old_cost_to_root[i])

    def test_grow_tree(self):
        x_goal = np.array([0, 0., 0.])
        dut = rrt_star.RrtStar(self.plant, 3, self.u_lo, self.u_up, x_goal)
        dut.grow_tree(10)
        self.assertEqual(dut.node_state.shape[0], 10)
        for i in range(dut.node_state.shape[0]):
            for child in dut.node_children[i]:
                self.assertEqual(dut.node_parent[child], i)
            if dut.node_parent[i] is None:
                np.testing.assert_allclose(dut.node_state[i], x_goal)
                self.assertEqual(dut.node_cost_to_root[i], 0)
            else:
                self.assertAlmostEqual(
                    dut.node_cost_to_root[dut.node_parent[i]] +
                    dut.node_cost_to_parent[i], dut.node_cost_to_root[i])
                np.testing.assert_allclose(dut.node_state[i],
                                           dut.node_to_parent_x[i][:, 0])
                np.testing.assert_allclose(dut.node_state[dut.node_parent[i]],
                                           dut.node_to_parent_x[i][:, -1])
                # Check the path connecting nodes.
                for j in range(dut.node_to_parent_x[i].shape[1] - 1):
                    xdot_l = dut.plant.dynamics(dut.node_to_parent_x[i][:, j],
                                                dut.node_to_parent_u[i][:, j])
                    xdot_r = dut.plant.dynamics(
                        dut.node_to_parent_x[i][:, j + 1],
                        dut.node_to_parent_u[i][:, j + 1])
                    np.testing.assert_allclose(
                        dut.node_to_parent_x[i][:, j + 1] -
                        dut.node_to_parent_x[i][:, j],
                        (xdot_l + xdot_r) * dut.node_to_parent_dt[i][j] / 2,
                        atol=1E-5)


if __name__ == "__main__":
    unittest.main()
