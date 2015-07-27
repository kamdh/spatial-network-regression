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
  int ny=400;
  int ninj=10;
  // setup other vars
  mat X=randu<mat>(nx,ninj);
  mat Y=randu<mat>(ny,ninj);
  mat W=randn<mat>(ny,nx);
  sp_mat Lx(nx,nx);
  sp_mat Ly(ny,ny);
  //double alpha=0.1;
  double lambda=1.0;

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
  for (int i=0; i<ny; i++) {
    Ly(i,i)=-2;
    if (i>0) {
      Ly(i,i-1) = 1;
    }
    if (i<(ny-1)) {
      Ly(i,i+1) = 1;
    }
  }
  
  //cout << "W0:\n" << W << endl;

  if (minimize_func(W,X,Y,Lx,Ly,lambda,200,1e10,1e-5) != 0) 
    return(1);

  //cout << "W final:\n" << W << endl;
  return(0);
}

/*
int test_copy(void) {
  long i,j;
  long nx=2;
  long ny=2;
  // initialize A, v
  mat A=randu<mat>(nx,ny);
  double *v;
  v=(double *) malloc(nx*ny*sizeof(double));
  // copy A into v
  copy_mat_2_vec(A,v);
  //display both
  cout << "A: " << A << endl;
  cout << "v: ";
  for (i=0; i<nx; i++) {
    for (j=0; j<ny; j++) {
      printf("%1.4f ",v[i*ny+j]);
    }
    printf("\n");
  }
  // change v, try copy to A
  v[3]=-10;
  copy_vec_2_mat(v,A);
  //display both
  cout << "A: " << A << endl;
  cout << "v: ";
  for (i=0; i<nx; i++) {
    for (j=0; j<ny; j++) {
      printf("%1.4f ",v[i*ny+j]);
    }
    printf("\n");
  }
  free(v);
  return(0);
}
*/
