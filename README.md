Spatial Network Regression
==========================

Fit a network connection matrix using smoothness regularized least-squares
regression. This solves the regularized regression problem:

```min_W>0 ||W*X-Y||^2 + lambda*(ninj/nx)*||W*Lx + Ly*W||^2```

The code implements checkpointing and can making an initial guess
by solving the unregularized problem.

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