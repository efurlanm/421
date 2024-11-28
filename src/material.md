Raissi et al. (2019) [[1](references.md#Raissi2019)] published an article about PINNs, which has 4152 citations. That work defines PINNs as ANNs trained to solve supervised learning tasks, but complying to physical laws, usually described by nonlinear PDEs. It also describes the use of ANNs to solve PDEs and obtain physics-informed surrogates of the physical model that are fully differentiable in all coordinates and free parameters. PINNs form a new class of data-efficient universal function approximators, which can be effectively trained using small datasets, and which may encode any underlying physical law.

Unlike standard numerical methods, the PINN solution can be obtained without specifying the spatial or temporal domain discretization. The training data is randomly sampled from simulations using synthetic data obtained using a known equation, or randomly generated, or even from observational data. This sampled data contains points in the space and time domain called collocation points (CPs). Except for the randomly generate data, provided that a sufficient number of CPs is available, a standard ANN may solve the PDE, otherwise a PINN would be required. A PINN uses a specific loss function in the training phase that embeds the applicable physical law and is calculated from the set of CPs and, eventually, also the ICs and BCs [[2](references.md#Cuomo2022)].

PINNs can be considered neural networks for supervised learning problems, as proposed here. However, PINNs can also be used as agents for reinforcement learning (RL) [[2](references.md#Cuomo2022)]. The most common PINN architectures are Multi-layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). Newer architectures are Auto-Encoder (AE), Deep Belief Network (DBN), Generative Adversarial Network (GAN) and Bayesian Deep Learning (BDL) [[2](references.md#Cuomo2022)]. 

The proposed test case requires the solution of a particular one-dimensional viscous Burger equation with Dirichlet boundary condition (BC) and initial condition (IC), which estimates the velocity field $u$ along time ([Equation&nbsp;1](#eq:burg)). Training data for the PINN is given by a set of CPs corresponding to the velocity field in different times are randomly generated within the considered domain.
 
In the train phase, the network then estimates a solution $u(t,x)$.
The function employed by the PINN $f(t,x)$ ([Equation&nbsp;2](#eq:ftx)) is derived from the known viscous Burgers equation, and allows to calculate the loss function.
In the following equations, $u$ is the velocity, the coefficient ${(100\pi)}^{-1}$ is the kinematic viscosity, and the subscripts denote partial differentiation in time and space, respectively, as
$u_t$ (which denotes $\frac{du}{dt}$),
$u_x$ (which denotes $\frac{du}{dx}$), and
$u_{xx}$ (which denotes $\frac{d^2u}{dx^2}$).

<span id='eq:burg'></span>
$$
u_t + uu_x - {(100\pi)}^{-1}u_{xx} = 0, \quad x \in [-1,1], \ t \in [0, 1],
\tag{1}
$$
$$ \nonumber u(0, x) = - sen(\pi x), \quad \ \ \text{(IC)} $$
$$ \nonumber u(t, -1) = u(t, 1) = 0. \quad \text{(BC)} $$

The viscous Burguers equation is employed to evaluate the error $f$ of the solution $u(t,x)$ estimated by the PINN, as shown in [Equation&nbsp;2](#eq:ftx).

<span id='eq:ftx'></span>
$$
f := u_t + uu_x - {(100\pi)}^{-1}u_{xx}
\tag{2}
$$

In this work, the PINN loss function to be minimized is given by the mean squared error ([Equation&nbsp;3](#eq:mse)) of two components, $MSE_u$, which embeds the error considering ICs and BCs, and $MSE_f$, which embeds the errors considering the set of CPs, where $t$ is the time step, and $x$ is the one-dimension coordinate. 

<span id='eq:mse'></span>
$$
MSE = MSE_u + MSE_f
\tag{3}
$$
where
$$ MSE_u = \frac{1}{N_u}\sum_{i=1}^{N_u}|u(t^i, x^i)-u^i|^2  \quad \text{(IC and BC)}$$
and
$$ MSE_f = \frac{1}{N_{CP}}\sum_{i=1}^{N_{CP}}|f(t^i, x^i)|^2  \quad \text{(CP)}$$

## A. Numerical GQM Implementation of the Test Problem

A Fortran 90 implementation of the GQM method was used to generate the full dataset of the velocity field, which was taken as reference solution in the comparison with the PINN solution.
The GQM dataset has 100 time steps in the interval [0, 99] and 256 one-dimensional grid points in the interval [-1, 1], defining a velocity field $u(x,t)$ subjected to the ICs and BCs shown in [Equation&nbsp;1](#eq:burg).

The GQM method is an iterative numerical algorithm that approximates the definite integral of a function as a weighted sum of the function values at specified points within the domain of integration [[3](references.md#Burkardt2013)]. The order of quadrature rule was set to 8. The loops corresponding to the compute-intensive part of the code were parallelized with the OpenMP 3.1 library using the ```!$OMP PARALLEL DO$``` directive, since there is no data dependency between loop iterations. The F90 code was compiled with GNU 4.8.5 setting the -O3 optimization flag. The code was also executed using CPU cores with 1, 4, 8, 16 and 24 OpenMP threads.


## B. PINN Implementation of the Test Problem

The particular architecture of the PINN implemented in this work is a feed-forward MLP with a 2-neuron input layer, eight 20-neuron hidden layers, and a single-neuron output layer. The loss function is the Mean Square Error (MSE). The minimization of the loss function is performed by an optimization method like the widespread Limited-memory BFGS (L-BFGS) algorithm, a quasi-Newton method that approximates the Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS). All hidden layers employ the hyperbolic tangent as the activation function.

The PINN implementation was based on the work of Raissi et al. (2019) [[1](references.md#Raissi2019)] and uses the [TensorFlow](http://www.tensorflow.org) 1.15 library and the Python 3.7 interpreter. Code snippets of the TensorFlow library are shown in [Listing&nbsp;1](#lst:utx) and [Listing&nbsp;2](#lst:ftx). Note that the snippets do not show the implementation of the BCs, ICs and other details. The code was also executed using CPU cores Tests with 1, 4, 8, 16 and 24 OpenMP threads.

<p id="lst:utx" style="font-size:.875em;text-align:center;">
Listing 1. Code snippet that implements $u(t,x)$.
</p>

```Python
def u(t, x):
    u = neural_net(tf.concat([t, x], 1), weights, biases)
    return u
```

<br>
<p id="lst:ftx" style="font-size:.875em;text-align:center;">
Listing 2. Code snippet that implements $f(t,x)$.
</p>

```Python
def f(t, x):
    u = u(t, x)
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    f = u_t + u*u_x - (0.01/tf.pi)*u_xx
    return f
```
