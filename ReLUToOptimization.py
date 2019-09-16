# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


def ComputeReLUActivationPath(model_relu, x):
    """
    For a given input x to a ReLU network, returns the activation path for
    this input.
    @param model_relu A ReLU network (constructed by Sequential Linear and
    ReLU units).
    @param x A numpy array, the input to the network.
    @return activation_path A list of length num_relu_layers.
    activation_path[i] is a list of 0/1, with size equal to the number of
    ReLU units in the i'th layer. If activation_path[i][j] = 1, then in the
    i'th layer, the j'th unit is activated.
    """
    activation_path = []
    layer_x = x
    for layer in model_relu:
        if (isinstance(layer, nn.Linear)):
            layer_x = layer.forward(layer_x)
        elif isinstance(layer, nn.ReLU):
            activation_path.append([relu_x >= 0 for relu_x in layer_x])
            layer_x = layer.forward(layer_x)
    return activation_path


def ReLUGivenActivationPath(model_relu, x_size, activation_path):
    """
    Given a ReLU network, and a given activation path, the ReLU network can
    be represented as
    ReLU(x) = gᵀx+h, while P*x ≤ q
    we return the quantity g, h, P and q.
    @param model_relu A ReLU network.
    @param x_size The size of the input x.
    @param activation_path A list of length num_relu_layers.
    activation_path[i] is a list of True/False, with size equal to the number
    of ReLU units in the i'th layer. If activation_path[i][j] = True, then in
    the i'th layer, the j'th unit is activated.
    @return (g, h, P, q)  g is a column vector, h is a scalar. P is a 2D
    matrix, and q is a column vector
    """
    A_layer = torch.eye(x_size)
    b_layer = torch.zeros((x_size, 1))
    relu_layer_count = 0
    num_linear_layer_output = None
    P = torch.empty((0, x_size))
    q = torch.empty((0, 1))
    for layer in model_relu:
        if (isinstance(layer, nn.Linear)):
            A_layer = layer.weight.data @ A_layer
            b_layer = layer.weight.data @ b_layer + \
                layer.bias.data.reshape((-1, 1))
            num_linear_layer_output = layer.weight.data.shape[0]
        elif (isinstance(layer, nn.ReLU)):
            assert(len(activation_path[relu_layer_count])
                   == num_linear_layer_output)
            for row, activation_flag in enumerate(
                    activation_path[relu_layer_count]):
                if activation_flag:
                    # If this row is active, then A.row(i) * x + b(i) >= 0
                    # is the same as -A.row(i) * x <= b(i)
                    P = torch.cat([P, -A_layer[row].reshape((1, -1))], dim=0)
                    q = torch.cat([q, b_layer[row].reshape((1, -1))], dim=0)
                else:
                    # If this row is inactive, then A.row(i) * x + b(i) <= 0
                    # is the same as A.row(i) * x <= -b(i)
                    P = torch.cat([P, A_layer[row].reshape((1, -1))], dim=0)
                    q = torch.cat([q, -b_layer[row].reshape((1, -1))], dim=0)
                    A_layer[row] = 0
                    b_layer[row] = 0
            relu_layer_count += 1
        else:
            raise Exception(
                "The ReLU network only allows ReLU and linear units.")
    g = A_layer
    h = b_layer
    return (g, h, P, q)
