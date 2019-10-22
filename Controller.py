# -*- coding: utf-8 -*-
import ReLUToOptimization
import ValueToOptimization

import cvxpy as cp
import numpy as np
import torch


class Controller:
    def __init__(self, model, value_fun, x_lo, x_up):
        self.model = model
        self.relu_opt = ReLUToOptimization.ReLUFreePattern(model, value_fun.dtype)
        self.relu_con = self.relu_opt.output_constraint(model, x_lo, x_up)
        
    def get_control_output(self, x0):
        # form an optimization problem
        # single step forward + approximated cost-to-go
        (Pin1, Pin2, Pin3, qrhs_in, 
        Peq1, Peq2, Peq3, qrhs_eq, 
        a_out, b_out, z_lo, z_up) = self.relu_con

        

        # recover n best cost to go
        
        # use the epsilon bounds to prune options, add the ones left to the queue
        
        # take the best candidate from the queue
        
        # fix ones more control action, solve the corresponding MIP
        pass