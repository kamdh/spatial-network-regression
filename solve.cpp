#include "functions.hpp"
#include <cstdio>
#include <cstdlib>
#include <csignal>

const string usage=
  "Usage: solve [W0] X Y Lx Ly lambda W\n\n"
  "Solve the regularized regression problems:\n"
  "  min_W>0 ||W*X-Y||^2 + lambda*(ninj/nx)*||W*Lx + Ly*W||^2\n\n"
  "By default, use initial guess W0=Y*X^T*pinv(X*X^T) "
  "for iterative solver.\n\n";

volatile sig_atomic_t checkpoint=0;

void catch_signal(int sig) {
  // This variable indicates that we need to run a checkpoint
  checkpoint=1;
}

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
  } else if (argc==7) {
    X_fn=argv[1];
    Y_fn=argv[2];
    Lx_fn=argv[3];
    Ly_fn=argv[4];
    if (sscanf(argv[5],"%lf",&lambda) != 1) {
      cout << "Error reading lambda" << endl;
      cout << usage;
      return(1);
    }
    outputfile=argv[6];
  } else {
    cout << "Error, number of input arguments is incorrect" << endl;
    cout << usage;
    return(1);
  }
  printf("Read lambda = %1.4e\n",lambda);
  mat W;  // current iterate
  printf("Reading X\n");
  mat X=arma_mat_mmread(X_fn);
  printf("Reading Y\n");
  mat Y=arma_mat_mmread(Y_fn);
  printf("Reading Lx\n");
  sp_mat Lx=arma_sp_mat_mmread(Lx_fn);
  printf("Reading Ly\n");
  sp_mat Ly=arma_sp_mat_mmread(Ly_fn);
  // Initialize W iterate
  printf("Initializing W0... ");
  if (argc==8) {
    W=arma_mat_mmread(W_fn);
    printf("successfully read W0.\n");
  } else if (argc==7) {
    // First try and load any (possible) checkpoint
    if (init_checkpoint(W, outputfile)) {
      // Checkpoint loading failed, initialize from start
      printf("performing pseudinverse initialization... ");
      init_W(W,X,Y);
      printf("done.\n");
    } else {
      printf("successfully loaded checkpoint.\n");
    }
  }

  // register signal handler for checkpointing
  signal(SIGTERM,catch_signal);
  signal(SIGINT,catch_signal);
  
  int code=minimize_func(W,X,Y,Lx,Ly,lambda,800,1e10,1e-5);
  // while (code==2) {
  //   cout << "minimize_func returned 2, restarting" << endl;
  //   code=minimize_func(W,X,Y,Lx,Ly,lambda,300,1e10,1e-5);
  // }
  if (code != 0) {
    cout << "Solver did not converge for call:" << endl;
    for (int n=0; n<argc; n++)
      cout << argv[n] << " ";
    cout << endl;
  }
  
  if (arma_mat_mmwrite(outputfile,W))
    return(1);
  
  return(0);
}
