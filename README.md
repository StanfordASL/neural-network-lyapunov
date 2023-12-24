# Introduction
This repo contains the code for the two papers
   - [Counter-example guided synthesis of neural network Lyapunov functions for piecewise linear systems](http://groups.csail.mit.edu/robotics-center/public_papers/Dai20.pdf) <br>
       Hongkai Dai, Benoit Landry, Marco Pavone and Russ Tedrake <br>
       IEEE Conference on Decision and Control, 2020 <br>
       [video](https://www.youtube.com/watch?v=A8Bpqb27DEE)
   - [Lyapunov-stable neural-network control](http://groups.csail.mit.edu/robotics-center/public_papers/Dai21.pdf) <br>
       Hongkai Dai, Benoit Landry, Lujie Yang, Marco Pavone and Russ Tedrake <br>
       Robotics: Science and Systems, 2021 <br>
       [video](https://www.youtube.com/watch?v=vCBdd-VuTwc)

We can synthesize neural-network controllers with Lyapunov stability guarantees. Namely for all the initial states within a certain region, the controller will drive the system from these initial states to converge to the goal state.

# Setup

## Python requirements
We use python 3 in this project. You could first install the packages in requirements.txt.

## Installation
Please run
```
pip install -e .
```
to install the package together with the dependencies.

## Setup gurobi
Please refer to https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python- on setting up Gurobi. Note that by default Gurobi ships with a limited license that has size limits on the optimization problem. Please refer to https://support.gurobi.com/hc/en-us/articles/360051597492 on using your own academic or commential license.

## Run a toy example
You could run
```
$ python3 neural_network_lyapunov/test/train_toy_system_controller_demo.py --dimension=1
```
This will synthesize a stabilizing controller with a Lyapunov function for a toy 1D system (TODO: add some visualization at the end of the demo). You should see that the error printed on the screen decreases to almost 0. (The code is non-deterministic, so if it doesn't converge to 0 in the first trial, you can re-run the demo and hopefully it converges in the second trial).

# Contributing to repo
## Linting
We use `flake8` to check if the python code follows PEP standard. Before submitting the PR, you could run
```
$ cd neural_network_lyapunov
$ flake8 ./
```
to check if there are any violations.

## Unit test
I am a strong believer of unit test. We strongly encourage to add tests to the functions in the PR.

