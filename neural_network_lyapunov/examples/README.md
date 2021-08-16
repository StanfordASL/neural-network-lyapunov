# Running examples

## Pendulum
One first example is the inverted pendulum. You can run
```
python3 pendulum/train_pendulum_demo.py
```
which trains a neural-network controller together with the neural-network Lyapunov function, which guarantees the Lyapunov stability condition in a small neighbourhood around the upright equilibrium θ=π, θ̇=0.
