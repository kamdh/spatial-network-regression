#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <csignal>

extern "C" 
{
#include "lbfgsb.h"
#include "mmio.h"
}

using namespace std;
using namespace arma;

extern volatile sig_atomic_t checkpoint_and_exit;

int init_unif(mat &W, const mat &X, const mat &Y);
int init_pinv(mat &W, const mat &X, const mat &Y);
int init_checkpoint(mat &W, char *fn);
double loss(const mat &W, const mat &X, const mat &Y);
double regularizer(const mat &W, const sp_mat &Lx, const sp_mat &Ly);
double cost(const mat &W, const mat &X, const mat &Y, 
            const sp_mat &Lx, const sp_mat &Ly, double lambda);
void gradient(mat &g,
              const mat &W, const sp_mat &Lx, const sp_mat &Ly, double lambda,
              const mat &YXT, const mat &XXT, const sp_mat &LxLxT, 
              const sp_mat &LyLyT);
void copy_vec_2_mat(double *v, mat &A);
void copy_mat_2_vec(const mat &A, double *v);
int arma_sp_mat_mmread(char *fn, sp_mat &M);
int arma_mat_mmread(char *fn, mat &M);
int arma_mat_mmwrite(char *fn, const mat &M);
int test_problem(void);
int test_copy(void);
int minimize_func(mat &W, const mat &X, const mat &Y, 
                  const sp_mat &Lx, const sp_mat &Ly, double lambda,
                  integer m,
                  int maxiter=200, double factr=1e7, double pgtol=1e-5,
                  int checkpt_iter=0, char *checkpt_file=NULL);

#endif
