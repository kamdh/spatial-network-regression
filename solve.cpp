#include "functions.hpp"
#include <cstdio>
#include <cstdlib>
#include <csignal>

const string usage=
  "Usage: solve [W0 [Omega]] X Y Lx Ly lambda W\n\n"
  "Solve the regularized regression problems:\n"
  "  min_W>0 ||W*X-Y||^2 + lambda*(ninj/nx)*||W*Lx + Ly*W||^2\n"
  "or, if Omega is given, solve:\n"
  "  min_W>0 ||P(W*X-Y)||^2 + lambda*(ninj/nx)*||W*Lx + Ly*W||^2\n"
  "where P=P_Omega^c is the projection onto the complement of Omega.\n\n"
  "By default, try to load W.CHECKPT, otherwise "
  "use initial guess W0 all zeros or supplied W0 "
  "for iterative solver "
  "(W0 can be --W0_init to initialize).\n\n";

const string header=
  "Regularized connectome regression\n"
  "=================================\n";

volatile sig_atomic_t checkpoint_and_exit=0;

void catch_signal(int sig) {
  // This variable indicates that we need to run a checkpoint
  checkpoint_and_exit=1;
}

int main(int argc, char** argv) {
  /*
    returns 0 if converged,
            1 if error,
            2 if checkpointed.
   */
  char *W_fn, *X_fn, *Y_fn, *Lx_fn, *Ly_fn, *Omega_fn, *outputfile;
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
  } else if (argc==9) {
    W_fn=argv[1];
    Omega_fn=argv[2];
    X_fn=argv[3];
    Y_fn=argv[4];
    Lx_fn=argv[5];
    Ly_fn=argv[6];
    if (sscanf(argv[7],"%lf",&lambda) != 1) {
      cout << "Error reading lambda" << endl;
      cout << usage;
      return(1);
    }
    outputfile=argv[8];
  } else {
    cout << "Error, number of input arguments is incorrect" << endl;
    cout << usage;
    return(1);
  }
  cout << header; 
  printf("Got %d arguments\n", argc);
  printf("Read lambda = %1.4e\n",lambda);
  // initialize
  mat W, X, Y;
  sp_mat Lx, Ly;
  printf("Reading X\n");
  if (load_matrix(X_fn, X))
    return(1);
  cout << "X.n_rows = " << X.n_rows << endl;
  cout << "X.n_cols = " << X.n_cols << endl;
  printf("Reading Y\n");
  if (load_matrix(Y_fn, Y))
    return(1);
  printf("Reading Lx\n");
  if (load_sparse_matrix(Lx_fn, Lx))
    return(1);
  printf("Reading Ly\n");
  if (load_sparse_matrix(Ly_fn, Ly))
    return(1);

  // Checkpoint filename is outputfile.CHECKPT
  char *checkpt_file;
  if (asprintf(&checkpt_file,"%s.CHECKPT", outputfile) == -1) {
    free(checkpt_file);
    return(1);
  }

  // Initialize W
  printf("Initializing W0... ");
  // Always try checkpoint first
  if (init_checkpoint(W,checkpt_file)) {
    if ( (argc==7) || (strcmp(W_fn,"--W0_init") == 0) ) {
      // Checkpoint loading failed, initialize from start
      printf("performing uniform initialization... ");
      init_unif(W,X,Y);
      //printf("Performing pseudoinverse initialization... ");
      //init_pinv(W,X,Y);
      printf("done.\n");
    } else if (argc==8) {
      if (load_matrix(W_fn,W))
        return(1);
      else
        printf("successfully read W0.\n");
    } else if (argc==9) {
      if (load_matrix(W_fn,W))
        return(1);
      else
        printf("successfully read W0.\n");
    }
  } else {
    printf("successfully loaded presumed checkpoint.\n");
  }
  // Load Omega
  sp_mat Omega(Y.n_rows,Y.n_cols);
  if (argc == 9) {
    printf("Loading Omega... ");
    if (load_sparse_matrix(Omega_fn, Omega))
      return(1);
    else
      printf("successfully read Omega.\n");
  }

  
  // register signal handler for checkpointing
  // signal(SIGTERM,catch_signal); // sent by scheduler (disabled)
  signal(SIGINT,catch_signal);  // ^C
  
  // low tolerance:
  int code=minimize_func(W,X,Y,Lx,Ly,Omega,lambda,6,600,2.0e10,1.0e-5,
                         10,checkpt_file);
  // high tolerance, easy problems:
  // int code=minimize_func(W,X,Y,Lx,Ly,Omega,lambda,12,2000,1e10,1.0e-5,
  //                        100,checkpt_file);
  // while (code==2) {
  //   cout << "minimize_func returned 2, restarting" << endl;
  //   code=minimize_func(W,X,Y,Lx,Ly,lambda,300,1e10,1e-5);
  // }
  if (code != 0) {
    cout << "Solver did not converge for call:" << endl;
    for (int n=0; n<argc; n++)
      cout << argv[n] << " ";
    cout << endl;
    cout << "Received error code " << code << endl;
    // save checkpoint file when not converged
    if (!save_matrix(checkpt_file,W))
      return(1);
  } else {
    // only save final output if converged
    if (!save_matrix(outputfile, W))
      return(1);
    cout << "Saved in file: " << outputfile << endl;
    remove(checkpt_file);
  }
  free(checkpt_file);
  return(code);
}
