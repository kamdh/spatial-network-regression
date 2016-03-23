Spatial Network Regression
==========================

Fit a network connection matrix using smoothness regularized least-squares
regression. This solves the regularized regression problem:

```min_W>0 ||W*X-Y||^2 + lambda*(ninj/nx)*||W*Lx + Ly*W||^2```

or, if Omega is given, solve:

```min_W>0 ||P(W*X-Y)||^2 + lambda*(ninj/nx)*||W*Lx + Ly*W||^2```,

where P=P_Omega^c is the projection onto the complement of Omega.

The code implements checkpointing and can making an initial guess
by solving the unregularized problem.

Requirements: 
* HDF5
* Armadillo C++ linear algebra library linked to HDF5 (http://arma.sourceforge.net)
* mmio.c and mmio.h (http://math.nist.gov/MatrixMarket/mmio-c.html)
* L-BFGS-B-C library (https://github.com/kharris/L-BFGS-B-C)
You will have to edit the Makefile to point to these libraries.

Optional requirements:
* optimized BLAS (MKL, OpenBLAS, etc.). This will require editing the Makefile
to point to your installed libraries. An example with this structure is in
`Makefile.hyak`.

Building: `make all`

To run a test case, try running `test`. The final lines should be:

```
At iterate   160, f(x)= 4.85e+01, ||proj grad||_infty = 1.26e-02
At iterate   161, f(x)= 4.85e+01, ||proj grad||_infty = 9.73e-03
           * * * 
Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value
           * * * 
   N    Tit   Tnf  Tnint  Skip  Nact      Projg        F
120000   161   165 154333     0 107113	9.73e-03 4.84880e+01
F(x) = 4.848795812e+01
22
Cauchy                time 1.322e+00 seconds.
Subspace minimization time 2.035e+00 seconds.
Line search           time 5.812e+00 seconds.
 Total User time 1.045e+01 seconds.
CONVERGED!
```