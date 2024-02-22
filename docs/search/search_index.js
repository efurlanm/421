var __index = {"config":{"lang":["en"],"separator":"[\\s\\-]+","pipeline":["stopWordFilter"]},"docs":[{"location":"index.html","title":"Home","text":"Solution of a One-Dimensional Viscous Burgers' Equation Using a Physics-Informed Neural Network and a Gaussian Quadrature Method  <p> Eduardo F. Miranda ORCID: 0000-0003-1200-794X </p> <p> Last edited: 2024-02-19 Repository: http://efurlanm.github.io/421/ </p> <p>Abstract. This work compares the solutions of a one-dimensional viscous Burgers\u2019 equation of a test problem using a Physics Informed Neural Network (PINN) and a numerical Gaussian Quadrature Method (GQM) method. The Burgers' equation is a partial differential equation (PDE) with derivatives in both space and time, which is commonly solved by a numerical method. However, recent works have proposed the solution by means of Artificial Neural Networks (ANNs). Since the number of sample/collocation points (in space and time) required for an efficient training of the ANN would be too high, PINNs were proposed to allow the use of less sample points by embedding the related equation of physics into the simulation. This work compares the solutions of the one-dimensional viscosity Burgers' equation for a test problem obtained by the PINN and by the GQM methods. Accuracy and required processing time of the solutions, both executed in the LNCC Santos Dumont supercomputer, are also presented.</p>"},{"location":"1.%20Introduction.html","title":"1. Introduction","text":"<p>Many simulations are mathematically modeled by partial differential equations (PDEs), which have derivatives in space and time. However, the coefficients of these derivatives are unknowns, and the PDEs are usually solved by a numerical method, like the Finite Difference Method (FDM) or the numerical Gaussian Quadrature Method (GQM). Recent works proposed to solve PDEs using Artificial Neural Networks (ANN), which are Machine Learning (ML) algorithms. The universal approximation theorem states that a neural network can approximate any continuous function, provided the network has a sufficient number of hidden layer and that employs non-linear activation functions. This approach requires to know a large set of sample points in space and time in order to perform the training of the neural network, and such sample points are named Collocation Points (CPs). Since the required number of CPs would be too high, Physics Informed Neural Networks (PINNs) were proposed to allow the use of less CPs by including in the ANN the underlying physical laws related to the simulation.</p> <p>This work compares the solutions of the viscous Burgers equation, a PDE with derivatives in both space and time, for a test problem, by a PINN and a GQM. This equation models the velocity of a viscous fluid, being a particular case of the Burgers equation for fluid mechanics. The corresponding PINN and GQM solutions are compared in terms of accuracy and processing time, both executed in the Santos Dumont supercomputer. Tests were executed in a Bull B710 processing node of the supercomputer Santos Dumont of the LNCC (National Laboratory of Scientific Computing). It has two Intel Xeon E5-2695v2 Ivy Bridge 2.4 GHz 12-core processors (total of 24 cores), and 64 GB of main memory.</p> <p>The solution of PDEs by PINNs is relatively recent and acquiring knowledge in such approach may be useful for solving PDEs in some specific modules of numerical models used at CPTEC/INPE for weather and climate forecast. </p> <p>The code is available at https://github.com/efurlanm/421/tree/main/project</p>"},{"location":"2.%20Material%20and%20methods.html","title":"2. Material and methods","text":"<p>Raissi et al. (2019) [1] published an article about PINNs, which has 4152 citations. That work defines PINNs as ANNs trained to solve supervised learning tasks, but complying to physical laws, usually described by nonlinear PDEs. It also describes the use of ANNs to solve PDEs and obtain physics-informed surrogates of the physical model that are fully differentiable in all coordinates and free parameters. PINNs form a new class of data-efficient universal function approximators, which can be effectively trained using small datasets, and which may encode any underlying physical law.</p> <p>Unlike standard numerical methods, the PINN solution can be obtained without specifying the spatial or temporal domain discretization. The training data is randomly sampled from simulations using synthetic data obtained using a known equation, or randomly generated, or even from observational data. This sampled data contains points in the space and time domain called collocation points (CPs). Except for the randomly generate data, provided that a sufficient number of CPs is available, a standard ANN may solve the PDE, otherwise a PINN would be required. A PINN uses a specific loss function in the training phase that embeds the applicable physical law and is calculated from the set of CPs and, eventually, also the ICs and BCs [2].</p> <p>PINNs can be considered neural networks for supervised learning problems, as proposed here. However, PINNs can also be used as agents for reinforcement learning (RL) [2]. The most common PINN architectures are Multi-layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). Newer architectures are Auto-Encoder (AE), Deep Belief Network (DBN), Generative Adversarial Network (GAN) and Bayesian Deep Learning (BDL) [2]. </p> <p>The proposed test case requires the solution of a particular one-dimensional viscous Burger equation with Dirichlet boundary condition (BC) and initial condition (IC), which estimates the velocity field \\(u\\) along time (Equation\u00a01). Training data for the PINN is given by a set of CPs corresponding to the velocity field in different times are randomly generated within the considered domain.</p> <p>In the train phase, the network then estimates a solution \\(u(t,x)\\). The function employed by the PINN \\(f(t,x)\\) (Equation\u00a02) is derived from the known viscous Burgers equation, and allows to calculate the loss function. In the following equations, \\(u\\) is the velocity, the coefficient \\({(100\\pi)}^{-1}\\) is the kinematic viscosity, and the subscripts denote partial differentiation in time and space, respectively, as \\(u_t\\) (which denotes \\(\\frac{du}{dt}\\)), \\(u_x\\) (which denotes \\(\\frac{du}{dx}\\)), and \\(u_{xx}\\) (which denotes \\(\\frac{d^2u}{dx^2}\\)).</p> <p> $$ u_t + uu_x - {(100\\pi)}^{-1}u_{xx} = 0, \\quad x \\in [-1,1], \\ t \\in [0, 1], \\tag{1} $$ $$ \\nonumber u(0, x) = - sen(\\pi x), \\quad \\ \\ \\text{(IC)} $$ $$ \\nonumber u(t, -1) = u(t, 1) = 0. \\quad \\text{(BC)} $$</p> <p>The viscous Burguers equation is employed to evaluate the error \\(f\\) of the solution \\(u(t,x)\\) estimated by the PINN, as shown in Equation\u00a02.</p> <p> $$ f := u_t + uu_x - {(100\\pi)}^{-1}u_{xx} \\tag{2} $$</p> <p>In this work, the PINN loss function to be minimized is given by the mean squared error (Equation\u00a03) of two components, \\(MSE_u\\), which embeds the error considering ICs and BCs, and \\(MSE_f\\), which embeds the errors considering the set of CPs, where \\(t\\) is the time step, and \\(x\\) is the one-dimension coordinate. </p> <p> $$ MSE = MSE_u + MSE_f \\tag{3} $$ where $$ MSE_u = \\frac{1}{N_u}\\sum_{i=1}^{N_u}|u(t^i, x^i)-u^i|^2  \\quad \\text{(IC and BC)}$$ and $$ MSE_f = \\frac{1}{N_{CP}}\\sum_{i=1}^{N_{CP}}|f(t^i, x^i)|^2  \\quad \\text{(CP)}$$</p>"},{"location":"2.%20Material%20and%20methods.html#a-numerical-gqm-implementation-of-the-test-problem","title":"A. Numerical GQM Implementation of the Test Problem","text":"<p>A Fortran 90 implementation of the GQM method was used to generate the full dataset of the velocity field, which was taken as reference solution in the comparison with the PINN solution. The GQM dataset has 100 time steps in the interval [0, 99] and 256 one-dimensional grid points in the interval [-1, 1], defining a velocity field \\(u(x,t)\\) subjected to the ICs and BCs shown in Equation\u00a01.</p> <p>The GQM method is an iterative numerical algorithm that approximates the definite integral of a function as a weighted sum of the function values at specified points within the domain of integration [3]. The order of quadrature rule was set to 8. The loops corresponding to the compute-intensive part of the code were parallelized with the OpenMP 3.1 library using the <code>!$OMP PARALLEL DO$</code> directive, since there is no data dependency between loop iterations. The F90 code was compiled with GNU 4.8.5 setting the -O3 optimization flag. The code was also executed using CPU cores with 1, 4, 8, 16 and 24 OpenMP threads.</p>"},{"location":"2.%20Material%20and%20methods.html#b-pinn-implementation-of-the-test-problem","title":"B. PINN Implementation of the Test Problem","text":"<p>The particular architecture of the PINN implemented in this work is a feed-forward MLP with a 2-neuron input layer, eight 20-neuron hidden layers, and a single-neuron output layer. The loss function is the Mean Square Error (MSE). The minimization of the loss function is performed by an optimization method like the widespread Limited-memory BFGS (L-BFGS) algorithm, a quasi-Newton method that approximates the Broyden\u2013Fletcher\u2013Goldfarb\u2013Shanno algorithm (BFGS). All hidden layers employ the hyperbolic tangent as the activation function.</p> <p>The PINN implementation was based on the work of Raissi et al. (2019) [1] and uses the TensorFlow 1.15 library and the Python 3.7 interpreter. Code snippets of the TensorFlow library are shown in Listing\u00a01 and Listing\u00a02. Note that the snippets do not show the implementation of the BCs, ICs and other details. The code was also executed using CPU cores Tests with 1, 4, 8, 16 and 24 OpenMP threads.</p> <p> Listing 1. Code snippet that implements $u(t,x)$. </p> <pre><code>def u(t, x):\n    u = neural_net(tf.concat([t, x], 1), weights, biases)\n    return u\n</code></pre> <p></p> <p> Listing 2. Code snippet that implements $f(t,x)$. </p> <pre><code>def f(t, x):\n    u = u(t, x)\n    u_t = tf.gradients(u, t)[0]\n    u_x = tf.gradients(u, x)[0]\n    u_xx = tf.gradients(u_x, x)[0]\n    f = u_t + u*u_x - (0.01/tf.pi)*u_xx\n    return f\n</code></pre>"},{"location":"3.%20Results.html","title":"3. Results","text":"<p>The PINN solution \\(u(t,x)\\) is shown in Figure\u00a01, with the time \\(t\\) in the horizontal axis  and the spatial coordinate \\(x\\) in the vertical axis. The red marks in the boundaries of the graph represent the 100 randomly assigned points (BC+IC) used for training. The 10,000 CPs randomly generated are not shown. The color scale refers to the velocity \\(u(x,t)\\). The dashed vertical lines refer to 2 specific snapshots (\\(t=0.25\\) and \\(t=0.75\\)). Figure\u00a02 shows the superimposed solutions for PINN and GQM for these 2 snapshots, which are quite equivalent.</p>  Figure 1. PINN solution for the velocity $u(t,x)$. The horizontal axis denotes time $t$, and the vertical axis, the coordinate $x$. The red marks in the boundaries of the graph represent the 100 randomly assigned points (BC+IC) used for training. The color scale refers to the velocity. The dashed vertical lines refer to 2 snapshots ($t=0.25$ and $t=0.75$).   Figure 2. Superimposed solutions for PINN and GQM for the $t=0.25$ and $t=0.75$ snapshots. PINN solution is labeled as Prediction (in orange), and GQM solution is labeled as Exact (in blue).  <p>Table\u00a01 shows the processing times for the PINN and GQM solutions. PINN time is splitted into training time (Train) and prediction time (Predict). The singe-thread runtime of the GQM implementation was taken as reference. In all cases, the GQM implementation achieved the best performance, i.e. required less processing times, presented better speedups and parallel efficiencies, even if considering only the PINN prediction time.</p>  Table 1. Processing times, speedups and parallel efficiencies for the PINN and GQM solutions for different numbers of OpenMP threads. The GQM single-thread execution time was taken as a reference, highlighted in blue. Best values are highlighted in red.  <p></p>  Figure 3. Processing times (seconds) in function of number of OpenMP threads for the GQM and PINN implementations. \"Train\" refer to the PINN training phase, while \"Predict\" refers to the PINN test/prediction phase (for convenience, times above 0.1 seconds are not shown).  <p></p>  Figure 4. Speedups in function of the number of OpenMP threads for the GQM and PINN implementations. The dotted line indicates the linear speedup. \"Train\" refer to the PINN training phase, while \"Predict\" refers to the PINN test/prediction phase.  <p></p>  Figure 5. Parallel efficiencies in function of the number of OpenMP threads for the GQM and PINN implementations. \"Train\" refer to the PINN training phase, while \"Predict\" refers to the PINN test/prediction phase."},{"location":"4.%20Conclusions.html","title":"4. Conclusions","text":"<p>This work compares the solutions of a one-dimensional viscous Burgers\u2019 equation of a test problem using a Physics Informed Neural Network (PINN) and a numerical Gaussian Quadrature Method (GQM) method. The Burgers' equation is a partial differential equation (PDE) with derivatives in both space and time, which is commonly solved by a numerical method, as the GQM. A comparison of the accuracy and required processing time of both solutions executed in the LNCC Santos Dumont supercomputer is also presented for different number of OpenMP threads using CPU cores. The GQM presented much lower processing times, and better speedups and parallel efficiencies. As future work, it is intended to exploit other PINN architectures and numerical methods, as well as taking advantage of GPU use, mainly for the PINN.</p>"},{"location":"References.html","title":"References","text":"<ol> <li> Lagaris, I. E., Likas, A., &amp; Fotiadis, D. I. (1998). Artificial Neural Networks for Solving Ordinary and Partial Differential Equations. IEEE Transactions on Neural Networks, 9(5), 987\u20131000. https://doi.org/10.1109/72.712178 </li> <li> Raissi, M., Perdikaris, P., &amp; Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686\u2013707. https://doi.org/10.1016/j.jcp.2018.10.045 </li> <li> Burkardt, J. (2013). Investigating Uncertain Parameters in the Burgers Equation. Mathematics Department, Ajou University, Suwon, Korea.  https://people.sc.fsu.edu/~jburkardt/presentations/burgers_2013_ajou.pdf </li> </ol>"}]}