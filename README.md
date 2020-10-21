# Setup

## Python requirements
We use python 3 in this project. You could first install the packages in requirements.txt.

## Install gurobi
Please download gurobi from https://www.gurobi.com/products/gurobi-optimizer/. We require at least gurobi 9.0. After downloading the software, please install its Python API by following https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html

To check your gurobi installation, type the following command in your terminal:
```
$ python3 -c "import gurobipy"
```
There should be no error thrown when executing the command.

## Setup environment variable
In the terminal, please run
```
$ python3 setup.py
```
It will print out the command to setup the environment variables. Execute that command in your terminal.

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

## Submitting a pull request
Currently this is a private repo. In order to trigger CI on a private repo, please send your branch to upstream `https://github.com/StanfordASL/neural-network-lyapunov` (not your own fork), and then submit a PR from the upstream.
