

class ValueFunction():
    """
    This is a template class to implement value function
    """
    def get_value_function(self):
        """
        returns a function that takes x and returns (cost-to-go, result)
        """
        raise(NotImplementedError)

    def get_differentiable_value_function(self):
        """
        return a function that takes x and returns
        (cost-to-go array, state trajectory).
        Where the first cost-to-go is differentiable wrt x
        """
        raise(NotImplementedError)

    def result_to_traj(self, result):
        """
        returns x_traj, u_traj, alpha_traj, dt_traj from result
        """
        raise(NotImplementedError)