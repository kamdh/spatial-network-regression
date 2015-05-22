#include "functions.hpp"

int main(int argc, char** argv) {
  /*
    Call with 
      ./solve W0 X Y Lx Ly lambda output
      ./solve X Y Lx Ly lambda output

    The second case sets W0=Y*X^T*pinv(X*X^T)
   */

  char *W_fn, *X_fn, *Y_fn, *Lx_fn, *Ly_fn, *outputfile;
  double lambda;
  mat W;
  if (argc==8) {
    W_fn=argv[1];
    X_fn=argv[2];
    Y_fn=argv[3];
    Lx_fn=argv[4];
    Ly_fn=argv[5];
    if (sscanf(argv[6],"%lf",&lambda) != 1) {
      cout << "Error reading lambda" << endl;
      return(1);
    }
    outputfile=argv[7];
  } 
  else if (argc==7) {
    X_fn=argv[1];
    Y_fn=argv[2];
    Lx_fn=argv[3];
    Ly_fn=argv[4];
    if (sscanf(argv[5],"%lf",&lambda) != 1) {
      cout << "Error reading lambda" << endl;
      return(1);
    }
    outputfile=argv[6];
  }
  else {
    cout << "Error, number of input arguments is incorrect" << endl;
    return(1);
  }
  mat X=arma_mat_mmread(X_fn);
  mat Y=arma_mat_mmread(Y_fn);
  sp_mat Lx=arma_sp_mat_mmread(Lx_fn);
  sp_mat Ly=arma_sp_mat_mmread(Ly_fn);
  outputfile=outputfile;
  if (argc==8)
    W=arma_mat_mmread(W_fn);
  else if (argc==7) {
    cout << "Performing pseudinverse initialization... ";
    W=Y*X.t()*pinv(X*X.t());
    cout << "done." << endl;
  }

  if (minimize_func(W,X,Y,Lx,Ly,lambda,150,1e10,1e-5)) 
    return(1);
  
  //cout << "final W:\n" << W << endl;

  if (arma_mat_mmwrite(outputfile,W))
    return(1);

  return(0);
}
