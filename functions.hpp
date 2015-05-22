#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" 
{
#include "lbfgsb.h"
#include "mmio.h"
}

using namespace std;
using namespace arma;


double cost(mat W, mat X, mat Y, sp_mat Lx, sp_mat Ly, double lambda);
mat gradient(mat W, sp_mat Lx, sp_mat Ly, double lambda,
             mat YXT, mat XXT, sp_mat LxLxT, sp_mat LyLyT);
void copy_vec_2_mat(double *v, mat &A);
void copy_mat_2_vec(mat A, double *v);
sp_mat arma_sp_mat_mmread(char *fn);
mat arma_mat_mmread(char *fn);
int arma_mat_mmwrite(char *fn, mat M);
int test_problem(void);
int test_copy(void);
int minimize_func(mat &W, const mat &X, const mat &Y, 
                  const sp_mat &Lx, const sp_mat &Ly, double lambda,
                  int maxiter=200, double factr=1e7, double pgtol=1e-5);


#endif
