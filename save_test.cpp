#include "functions.hpp"

volatile sig_atomic_t checkpoint_and_exit=0;

int main(void) {
  arma_rng::set_seed(12588); // reproducibility
  // small defaults
  // int nx=3;
  // int ny=3;
  // int ninj=2;
  // big defaults
  int nx=300;
  int ny_i=400;
  int ny_c=400;
  int ninj=10;
  // setup other vars
  mat X=randu<mat>(nx,ninj);
  mat Y_i=randu<mat>(ny_i,ninj);
  mat W=randn<mat>(ny_i+ny_c,nx);
  mat Y_c=randu<mat>(ny_c,ninj);
  sp_mat Lx(nx,nx);
  sp_mat Ly_i(ny_i,ny_i);
  sp_mat Ly_c(ny_c,ny_c);

  // fill Laplacians
  for (int i=0; i<nx; i++) {
    Lx(i,i)=-2;
    if (i>0) {
      Lx(i,i-1) = 1;
    }
    if (i<(nx-1)) {
      Lx(i,i+1) = 1;
    }
  }
  for (int i=0; i<ny_i; i++) {
    Ly_i(i,i)=-2;
    if (i>0) {
      Ly_i(i,i-1) = 1;
    }
    if (i<(ny_i-1)) {
      Ly_i(i,i+1) = 1;
    }
  }
  for (int i=0; i<ny_c; i++) {
    Ly_c(i,i)=-2;
    if (i>0) {
      Ly_c(i,i-1) = 1;
    }
    if (i<(ny_c-1)) {
      Ly_c(i,i+1) = 1;
    }
  }

  arma_mat_mmwrite("test_X.mtx",X);
  arma_mat_mmwrite("test_Y_ipsi.mtx",Y_i);
  arma_mat_mmwrite("test_Y_contra.mtx",Y_c);
  return(0);
}
