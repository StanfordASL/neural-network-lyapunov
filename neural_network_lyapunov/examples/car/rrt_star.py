import neural_network_lyapunov.examples.car.unicycle_traj_opt as\
    unicycle_traj_opt
import neural_network_lyapunov.utils as utils
import numpy as np
import torch
import scipy.integrate
import queue
import pydrake.solvers.mathematicalprogram as mp


class RrtStar:
    def __init__(self, plant, x_dim, u_lo: np.ndarray, u_up: np.ndarray,
                 x_goal: np.ndarray):
        self.plant = plant
        self.x_dim = x_dim
        self.u_lo = u_lo
        self.u_up = u_up
        # node_state[i] is the state of node[i]
        self.node_state = np.empty((0, self.x_dim))
        # node_parent[i] is the index of the parent node of node i.
        self.node_parent = []
        # node_children[i] contains the indices of all the child nodes of node
        # i.
        self.node_children = []
        # node_cost_to_root[i] is the cost to the goal from node[i]
        self.node_cost_to_root = np.empty((0, ))
        # node_cost_to_parent[i] is the cost from node[i] to its parent.
        self.node_cost_to_parent = np.empty((0, ))

        # Information about the path from a node to its parent.
        self.node_to_parent_x = []
        self.node_to_parent_u = []
        self.node_to_parent_dt = []

        self._add_goal(x_goal)

    def _add_node(self, x_node: np.ndarray, parent_idx, cost_to_parent):
        return self._add_node_with_path(x_node, parent_idx, cost_to_parent,
                                        None, None, None)

    def _add_node_with_path(self, x_node: np.ndarray, parent_idx,
                            cost_to_parent, path_to_parent_x: np.ndarray,
                            path_to_parent_u: np.ndarray,
                            path_to_parent_dt: np.ndarray):
        """
        Add a node with the path (x, u, dt) from this node to the parent.
        """
        assert (isinstance(x_node, np.ndarray))
        assert (x_node.shape == (self.x_dim, ))
        self.node_state = np.append(self.node_state,
                                    x_node.reshape((1, self.x_dim)),
                                    axis=0)
        self.node_parent.append(parent_idx)
        self.node_children[parent_idx].add(self.node_state.shape[0] - 1)
        self.node_children.append(set())
        self.node_cost_to_root = np.append(
            self.node_cost_to_root,
            cost_to_parent + self.node_cost_to_root[parent_idx])
        self.node_cost_to_parent = np.append(self.node_cost_to_parent,
                                             cost_to_parent)
        self.node_to_parent_x.append(path_to_parent_x)
        self.node_to_parent_u.append(path_to_parent_u)
        self.node_to_parent_dt.append(path_to_parent_dt)
        return self.node_state.shape[0] - 1

    def _add_goal(self, x_goal: np.ndarray):
        assert (isinstance(x_goal, np.ndarray))
        assert (x_goal.shape == (self.x_dim, ))
        self.node_state = np.append(self.node_state,
                                    x_goal.reshape((1, self.x_dim)),
                                    axis=0)
        self.node_parent.append(None)
        self.node_children.append(set())
        self.node_cost_to_root = np.append(self.node_cost_to_root, 0)
        self.node_cost_to_parent = np.append(self.node_cost_to_parent, 0.)
        self.node_to_parent_x.append(None)
        self.node_to_parent_u.append(None)
        self.node_to_parent_dt.append(None)

    def state_distance(self, x1, x2):
        Q = np.diag([1., 1., 0.5])
        assert (x1.shape[-1] == self.x_dim)
        assert (x2.shape[-1] == self.x_dim)
        return np.sum((Q @ (x1 - x2).T).T * (x1 - x2), axis=-1)

    def nearest_node(self, x: np.ndarray):
        """
        Returns the index of the node with the nearest distance to x, measured
        as the weighted L2 distance with weights Q.
        """
        assert (isinstance(x, np.ndarray))
        assert (x.shape == (self.x_dim, ))
        nearest_node_idx = np.argmin(self.state_distance(self.node_state, x))
        return nearest_node_idx

    def extend_node(self, node_idx, num_samples, dt, x):
        """
        From a given node, try to forward simulate the states with many sampled
        control actions for time dt, and then return the state with the closest
        distance to x at the end of the simulation.
        """
        u_samples = utils.uniform_sample_in_box(torch.from_numpy(self.u_lo),
                                                torch.from_numpy(self.u_up),
                                                num_samples).detach().numpy()
        x_sim = []
        for i in range(num_samples):
            result = scipy.integrate.solve_ivp(
                lambda t, x: self.plant.dynamics(x, u_samples[i]), (0, dt),
                self.node_state[node_idx])
            x_sim.append(result.y[:, -1])
        x_sim = np.vstack(x_sim)
        distances = self.state_distance(x_sim, x)
        nearest_x_sim_idx = np.argmin(distances)
        x_extended = x_sim[nearest_x_sim_idx]
        return x_extended

    def find_path(self, x1, x2):
        """
        Find the path between x1 and x2.
        We solve a nonlinear optimization problem.
        return the cost of the path.
        """
        nT = 4
        dt_min = 0.
        dt_max = 0.02
        prog, initial_val_constraint, final_val_constraint, x, u, dt =\
            unicycle_traj_opt.construct_traj_opt(
                nT, self.u_lo, self.u_up, dt_min, dt_max)
        initial_val_constraint.evaluator().set_bounds(x1, x1)
        final_val_constraint.evaluator().set_bounds(x2, x2)
        prog.SetInitialGuess(x, np.linspace(x1, x2, nT).T)
        prog.SetInitialGuess(dt, dt_max * np.ones((nT - 1, )))
        result = mp.Solve(prog)
        if result.is_success():
            return result.get_optimal_cost(), result.GetSolution(
                x), result.GetSolution(u), result.GetSolution(dt)
        else:
            return np.inf,

    def neighbours(self, x, radius):
        """
        Return the indices of the nodes within distance radius to x.
        """
        distances = self.state_distance(self.node_state, x)
        in_sphere_flag = distances <= radius
        indices = np.arange(self.node_state.shape[0])[in_sphere_flag]
        return list(indices)

    def update_parent(self, node_idx, parent_idx, cost_to_parent,
                      path_to_parent_x, path_to_parent_u, path_to_parent_dt):
        """
        updates the parent of node[node_idx] to parent_idx.
        """
        old_parent = self.node_parent[node_idx]
        self.node_children[old_parent].remove(node_idx)
        self.node_parent[node_idx] = parent_idx
        self.node_children[parent_idx].add(node_idx)
        self.node_cost_to_parent[node_idx] = cost_to_parent
        self.node_cost_to_root[
            node_idx] = self.node_cost_to_root[parent_idx] + cost_to_parent
        self.node_to_parent_x[node_idx] = path_to_parent_x
        self.node_to_parent_u[node_idx] = path_to_parent_u
        self.node_to_parent_dt[node_idx] = path_to_parent_dt
        # Now update the costs of all the descent nodes.
        descent_queue = queue.Queue()
        descent_queue.put(self.node_children[node_idx])
        while not descent_queue.empty():
            children = descent_queue.get()
            for child in children:
                self.node_cost_to_root[child] = self.node_cost_to_root[
                    self.node_parent[child]] + self.node_cost_to_parent[child]
                descent_queue.put(self.node_children[child])

    def connect_state_extend(self, x_extend, neighbour_indices: list):
        """
        First try to connect x_extend to each node in the neighbours, find the
        connection with the smallest cost-to-root from x_extend, add the new
        node at x_extend as the child to the neighour node with the smallest
        cost-to-root.
        Then rewire all the neighbour nodes to the new node if the cost is
        smaller.
        """
        cost_to_neighbour = np.empty((len(neighbour_indices)))
        path_to_neighbour = [None] * len(neighbour_indices)
        for i in range(len(neighbour_indices)):
            path_to_neighbour[i] = self.find_path(
                x_extend, self.node_state[neighbour_indices[i]])
            cost_to_neighbour[i] = path_to_neighbour[i][0]
        if np.any(np.isfinite(cost_to_neighbour)):
            best_neighbour_index = np.argmin(
                cost_to_neighbour + self.node_cost_to_root[neighbour_indices])
            parent = neighbour_indices[best_neighbour_index]
            new_node_cost_to_root = cost_to_neighbour[
                best_neighbour_index] + self.node_cost_to_root[parent]
            new_node_idx = self._add_node_with_path(
                x_extend, parent, cost_to_neighbour[best_neighbour_index],
                path_to_neighbour[best_neighbour_index][1],
                path_to_neighbour[best_neighbour_index][2],
                path_to_neighbour[best_neighbour_index][3])
            # rewire if necessary.
            for i in range(len(neighbour_indices)):
                if cost_to_neighbour[
                        i] + new_node_cost_to_root < self.node_cost_to_root[
                            neighbour_indices[i]]:
                    self.update_parent(neighbour_indices[i], new_node_idx,
                                       cost_to_neighbour[i],
                                       np.fliplr(path_to_neighbour[i][1]),
                                       np.fliplr(path_to_neighbour[i][2]),
                                       path_to_neighbour[i][3][::-1].copy())
            return new_node_idx
        else:
            return None

    def sample_state(self):
        x_lo = np.array([-5, -5, -1.2 * np.pi])
        x_up = np.array([5., 5., 1.2 * np.pi])
        x = np.empty((3, ))

        for i in range(3):
            x[i] = np.random.uniform(x_lo[i], x_up[i])
        return x

    def grow_tree(self, max_nodes):
        while self.node_state.shape[0] < max_nodes:
            x_sample = self.sample_state()
            nearest_node = self.nearest_node(x_sample)
            x_extend = self.extend_node(nearest_node,
                                        num_samples=20,
                                        dt=0.05,
                                        x=x_sample)
            radius = 0.2
            neighbour_nodes = self.neighbours(x_extend, radius)
            if len(neighbour_nodes) > 0:
                self.connect_state_extend(x_extend, neighbour_nodes)

    def save_tree(self, file_path):
        """
        Save the data to a file
        """
        torch.save(
            {
                "node_state": self.node_state,
                "node_parent": self.node_parent,
                "node_children": self.node_children,
                "node_cost_to_root": self.node_cost_to_root,
                "node_cost_to_parent": self.node_cost_to_parent,
                "node_to_parent_x": self.node_to_parent_x,
                "node_to_parent_u": self.node_to_parent_u,
                "node_to_parent_dt": self.node_to_parent_dt
            }, file_path)

    def load_tree(self, file_path):
        data = torch.load(file_path)
        self.node_state = data["node_state"]
        self.node_parent = data["node_parent"]
        self.node_children = data["node_children"]
        self.node_cost_to_root = data["node_cost_to_root"]
        self.node_cost_to_parent = data["node_cost_to_parent"]
        self.node_to_parent_x = data["node_to_parent_x"]
        self.node_to_parent_u = data["node_to_parent_u"]
        self.node_to_parent_dt = data["node_to_parent_dt"]
