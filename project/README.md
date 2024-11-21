# Course project - CAP-421

This work compares the solutions of a one-dimensional viscous Burgers’ equation of a test problem using a Physics Informed Neural Network (PINN) and a numerical Gaussian Quadrature Method (GQM) method. The Burgers' equation is a partial differential equation (PDE) with derivatives in both space and time, which is commonly solved by a numerical method. However, recent works have proposed the solution by means of Artificial Neural Networks (ANNs). Since the number of sample/collocation points (in space and time) required for an efficient training of the ANN would be too high, PINNs were proposed to allow the use of less sample points by embedding the related equation of physics into the simulation. This work compares the solutions of the one-dimensional viscosity Burgers' equation for a test problem obtained by the PINN and by the GQM methods. Accuracy and required processing time of the solutions, both executed in the LNCC Santos Dumont supercomputer, are also presented.


- [Manuscript LaTeX sources](manuscript) directory - manuscript of the project developed in the course.

- Manuscript: DOI [10.5281/zenodo.10676900](https://doi.org/10.5281/zenodo.10676900).

- [Online HTML version of the manuscript](https://efurlanm.github.io/421/).

- [PDF version of the manuscript](Preprint-PINN-GQM.pdf).
