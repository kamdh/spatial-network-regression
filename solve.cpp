#include "functions.hpp"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>


const string usage=
  "Usage: solve [W0] X Y Lx Ly lambda W\n\n"
  "Solve the regularized regression problems:\n"
  "  min_W>0 ||W*X-Y||^2 + lambda^2*||W*Lx + Ly*W||^2\n\n"
  "By default, use initial guess W0=Y*X^T*pinv(X*X^T) "
  "for iterative solver.\n\n";

int main(int argc, char** argv) {
  char *W_fn, *X_fn, *Y_fn, *Lx_fn, *Ly_fn, *outputfile;
  double lambda;

  // Parse arguments
  if (argc==8) {
    W_fn=argv[1];
    X_fn=argv[2];
    Y_fn=argv[3];
    Lx_fn=argv[4];
    Ly_fn=argv[5];
    if (sscanf(argv[6],"%lf",&lambda) != 1) {
      cout << "Error reading lambda" << endl;
      cout << usage;
      return(1);
    }
    outputfile=argv[7];
  } 
  else if (argc==7) {
    X_fn=argv[1];
    Y_fn=argv[2];
    Lx_fn=argv[3];
    Ly_fn=argv[4];
    outputfile=argv[6];
  }
  else {
    cout << "Error, number of input arguments is incorrect" << endl;
    cout << usage;
    return(1);
  }
  mat W;  // current iterate
  mat X=arma_mat_mmread(X_fn);
  mat Y=arma_mat_mmread(Y_fn);
  sp_mat Lx=arma_sp_mat_mmread(Lx_fn);
  sp_mat Ly=arma_sp_mat_mmread(Ly_fn);
  outputfile=outputfile;
  // Initialize W iterate
  if (argc==8)
    W=arma_mat_mmread(W_fn);
  else if (argc==7) {
    cout << "Performing pseudinverse initialization... ";
    W=Y*X.t()*pinv(X*X.t());
    cout << "done." << endl;
  }
  
  int code=minimize_func(W,X,Y,Lx,Ly,lambda,600,1e10,1e-5);
  // while (code==2) {
  //   cout << "minimize_func returned 2, restarting" << endl;
  //   code=minimize_func(W,X,Y,Lx,Ly,lambda,300,1e10,1e-5);
  // }
  if (code==2) {
    cout << "Solver did not converge for call:" << endl;
    for (int n=0; n<argc; n++)
      cout << argv[n] << " ";
    cout << endl;
  }

  if (arma_mat_mmwrite(outputfile,W))
    return(1);

  return(0);
}
