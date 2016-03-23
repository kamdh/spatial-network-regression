#include "functions.hpp"

int init_unif(mat &W, const mat &X, const mat &Y) {
  W.zeros(Y.n_rows, X.n_rows);
  //W.ones(Y.n_rows, X.n_rows);
  return(0);
}

int init_pinv(mat &W, const mat &X, const mat &Y) {
  //W=Y*X.t()*pinv(X*X.t());
  double tol;
  mat U,V,pinvXXt;
  vec s,sthresh,s2inv;
  if (!svd_econ(U,s,V,X,"left")) {
    cout << "SVD failed in W0 initialization" << endl;
    return(1);
  }
  tol=U.n_rows*max(s)*datum::eps;
  sthresh=s(s > tol);
  s2inv=1/(sthresh%sthresh);
  pinvXXt=U*diagmat(s2inv)*U.t();
  W=Y*X.t()*pinvXXt;
  return(0);
}

int init_checkpoint(mat &W, const char *fn) {
  FILE* fptr = fopen(fn, "r");
  if (fptr != NULL) {
    fclose(fptr);
    // actually load it now
    if (load_matrix(fn,W))
      return(1);
    else
      return(0);
  } else {
    // file cannot be opened
    return(1);
  }
}

mat P_Omega(const mat &A, const sp_mat &Omega) {
  mat PA = A;
  sp_mat::const_iterator start = Omega.begin();
  sp_mat::const_iterator end   = Omega.end();
  
  for(sp_mat::const_iterator it = start; it != end; ++it) {
    // cout << "location: " << it.row() << "," << it.col() << "  ";
    // cout << "value: " << (*it) << endl;
    PA(it.row(),it.col()) = 0.0;
  }
  //  PA.elem( Omega.find() ).zeros();
  return(PA);
}

double loss(const mat &W, const mat &X, const mat &Y, const sp_mat &Omega) {
  /* || Y - W*X ||_F^2  */
  if ( !Omega.is_empty() ) {
    return pow(norm(P_Omega(Y-W*X,Omega),"fro"),2);
  } else {
    return pow(norm(Y-W*X,"fro"),2);
  }
}

double regularizer(const mat &W, const sp_mat &Lx, const sp_mat &Ly) {
  /* || W*Lx' + Ly*W ||_F^2 */
  return pow(norm(W*Lx.t() + Ly*W,"fro"),2);
}

double cost(const mat &W, const mat &X, const mat &Y,
            const sp_mat &Lx, const sp_mat &Ly, 
            const sp_mat &Omega, double lambda) {
  /* Returns the cost of the current parameters */
  double cost1, cost2;
  cost1 = loss(W,X,Y,Omega);
  cost2 = regularizer(W,Lx,Ly);
  return cost1+lambda*cost2;
}

void gradient(mat &g,
              const mat &W, const mat &X, const mat &Y,
              const sp_mat &Lx, const sp_mat &Ly, 
              const sp_mat &Omega,
              double lambda,
              const mat &YXT, const mat &XXT, 
              const sp_mat &LxLxT, const sp_mat &LyLyT) {
  /* Computes the gradient g, which is modified in-place */
  if ( !Omega.is_empty() ) {
    g=2*P_Omega(W*X-Y,Omega)*X.t() + 2*lambda*(2*Ly*W*Lx + W*LxLxT + LyLyT*W);
  } else {
    g=-2*YXT + 2*W*XXT + 2*lambda*(2*Ly*W*Lx + W*LxLxT + LyLyT*W);
  }
}

void copy_vec_2_mat(double *v, mat &A) {
  long m=(long) A.n_rows;
  long n=(long) A.n_cols;
  long i,j;
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      A(i,j)=v[i*n+j]; // could access memory outside for incompatible A,v
    }
  }
}

void copy_mat_2_vec(mat &A, double *v) {
  long m=(long) A.n_rows;
  long n=(long) A.n_cols;
  long i,j;
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v[i*n+j]=A(i,j); // could cause buffer overflow if v is wrong size
    }
  }
}

int load_matrix(const char *fn, mat &M) {
  // first try arma built-in load
  if (!M.load(fn)) {
    // second try mmread
    if (arma_mat_mmread(fn, M)) {
      // all failed
      return(1);
    }
  } else {
    // M.load returns transpose
    inplace_trans(M);
  }
  // one succeeded
  return(0);
}

int load_sparse_matrix(const char *fn, sp_mat &M) {
  // // first try arma built-in load
  // if (!M.load(fn)) {
  //   // if failed, try sparse version
  if (arma_sp_mat_mmread(fn,M)) {
    return(1);
  }
  // one succeeded
  return(0);
}

int save_matrix(const char *fn, const mat &M) {
  mat Mt = M.t();
  return(Mt.save(fn, hdf5_binary));
}

int arma_sp_mat_mmread(const char *fn, sp_mat &M) {
  /*
    load a matrix market sparse matrix into arma sp_mat
    
    adapted from example_read.c from 
   */
    MM_typecode matcode;
    FILE *f;
    int ret_code;
    int m, n, nz;
    int I, J;
    double val;

    if ((f = fopen(fn, "r")) == NULL) {
      printf("Error opening file %s\n", fn);
      return(1);
    }

    if (mm_read_banner(f, &matcode) != 0) {
      printf("Could not process Matrix Market banner.\n");
      return(1);
    }

    if (!(mm_is_matrix(matcode)) || mm_is_dense(matcode)
        || mm_is_complex(matcode)) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return(1);
    }

    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nz)) !=0) {
      printf("Error reading matrix header\n");
      return(1);
    }

    // initialize matrices for insertion into M
    umat locations(2,nz);
    vec values(nz);

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    for (int i=0; i<nz; i++) {
      fscanf(f, "%d %d %lg\n", &I, &J, &val);
      I--;  // adjust from 1-based to 0-based
      J--;
      locations(0,i)=I;
      locations(1,i)=J;
      values(i)=val;
    }

    if (f !=stdin) fclose(f);

    // now construct sparse arma matrix
    
    if (mm_is_symmetric(matcode)) {
      sp_mat Ml(locations,values,m,n);
      M=Ml+Ml.t();
      M.diag() /= 2;
      return(0);
    } else {
      sp_mat Minit(locations,values,m,n);
      M=Minit;
      return(0);
    }
}

int arma_mat_mmread(const char *fn, mat &M) {
  /*
    load a matrix market matrix into arma mat
    
    adapted from example_read.c from 
   */
    MM_typecode matcode;
    FILE *f;
    int m, n;
    int ret_code;
    double val;

    if ((f = fopen(fn, "r")) == NULL) {
      printf("Error opening file %s\n", fn);
      return(1);
    }

    if (mm_read_banner(f, &matcode) != 0) {
      printf("Could not process Matrix Market banner.\n");
      return(1);
    }

    if (!(mm_is_matrix(matcode)) || mm_is_sparse(matcode) || 
        mm_is_complex(matcode)) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return(1);
    }

    /* find out size of matrix .... */
    if ((ret_code = mm_read_mtx_array_size(f, &m, &n)) !=0) {
      printf("Error reading matrix header\n");
      return(1);
    }

    M.set_size(m, n);        

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    for (int j=0; j<n; j++) {
      for (int i=0; i<m; i++) { // column-major: loop over rows fastest
        fscanf(f, "%lg\n", &val);
        M(i,j)=val;
      }
    }

    if (f !=stdin) fclose(f);

    if (mm_is_symmetric(matcode)) {
      // M=M+M.t();
      // M.diag() /= 2;
      M=symmatl(M);
    } 
    return(0);
}

int arma_mat_mmwrite(const char *fn, const mat &M) {
  /*
    save an arma mat in matrix market format
    
    adapted from example_write.c
   */
  MM_typecode matcode;
  int i,j;
  FILE *f;

  int m = M.n_rows;
  int n = M.n_cols;

  if ((f = fopen(fn, "w")) == NULL) {
    printf("Error opening file %s\n", fn);
    exit(1);
  }
  
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_array(&matcode);
  mm_set_real(&matcode);

  mm_write_banner(f, matcode); 
  mm_write_mtx_array_size(f, m, n);
  
  for (j=0; j<n; j++)
    for (i=0; i<m; i++) // column-major: loop over rows fastest
      fprintf(f, "%16g\n", M(i,j));

  if (f != stdout) fclose(f);

  return(0);
}

int minimize_func(mat &W, const mat &X, const mat &Y, 
                  const sp_mat &Lx, const sp_mat &Ly, 
                  const sp_mat &Omega,
                  double lambda,
                  integer m,
                  int maxiter, double factr, double pgtol,
                  int checkpt_iter, char *checkpt_file) {
  // Check dimensions
  uint nx=W.n_cols;
  uint ny=W.n_rows;
  uint ninj=X.n_cols;
  if ((nx != X.n_rows) || (nx != Lx.n_rows) || !(Lx.is_square())) {
    printf("Matrix dimension nx inconsistent\n");
    return(1);
  }
  if ((ny != Y.n_rows) || (ny != Ly.n_rows) || !(Ly.is_square())) { 
    printf("Matrix dimension ny inconsistent\n");
    return(1);
  }
  if ((ninj != X.n_cols) || (ninj != Y.n_cols)) {
    printf("Matrix dimension ninj inconsistent\n");
    return(1);
  }
  if ( !Omega.is_empty() && ((ny != Omega.n_rows) || (ninj != Omega.n_cols)) ) {
    printf("Dimensions of Omega inconsistent\n");
    return(1);
  }
  if (lambda <= 0.0) {
    printf("lambda should be > 0\n");
    return(1);
  }
  // Scale lambda
  lambda=lambda*double(ninj)/double(nx);
  
  // Do a few computations to setup matrix products for gradient
  mat YXT,XXT;
  sp_mat LxLxT,LyLyT;
  YXT=Y*X.t();
  XXT=X*X.t();
  LxLxT=Lx*Lx.t();
  LyLyT=Ly*Ly.t();
  // Initial test
  double f;
  mat gmat(ny,nx);
  // f=cost(W,X,Y,Lx,Ly,lambda);
  // cout << "initial cost: " << f << endl;
  // gradient(gmat,W,Lx,Ly,lambda,YXT,XXT,LxLxT,LyLyT);
  // cout << "initial gradient norm: " << norm(gmat,"inf") << endl;
  
  // Run L-BFGS-B optimization
  /* Local variables */
  int retcode; // code to return
  int converged = 0;
  double *g;
  integer i;
  double *l,*u;
  integer n;
  double *x;
  //double t1, t2;
  double *wa;
  integer *nbd, *iwa;
  integer taskValue;
  integer *task=&taskValue; /* must initialize !! */
  /*      http://stackoverflow.com/a/11278093/269192 */
  //double factr;
  integer csaveValue;
  integer *csave=&csaveValue;
  double dsave[29];
  integer isave[44];
  logical lsave[4];
  //double pgtol;
  integer iprint;
  
  // output at every iteration
  iprint = 1; 
  // tolerances in the stopping criteria 
  //factr = 1e7;
  //pgtol = 1e-5;
  /*     We specify the dimension n of the sample problem and the number */
  /*     m of limited memory corrections stored.  (n and m should not */
  /*     exceed the limits nmax and mmax respectively.) */
  n = nx*ny;
  //m = 5;
  // allocate vectors
  x=(double *) malloc(n*sizeof(double));
  g=(double *) malloc(n*sizeof(double));
  u=(double *) malloc(n*sizeof(double));
  l=(double *) malloc(n*sizeof(double));
  nbd=(integer *) malloc(n*sizeof(integer));
  // wa is a double precision working array of length 
  //   (2mmax + 5)nmax + 12mmax^2 + 12mmax. 
  wa=(double *) malloc(((2*m+5)*n+12*m*m+12*m)*sizeof(double));
  // iwa is an INTEGER array of length 3nmax
  iwa=(integer *) malloc(3*n*sizeof(integer));
  /*     We now provide nbd which defines the bounds on the variables: */
  /*                    l   specifies the lower bounds, */
  /*                    u   specifies the upper bounds. */
  for (i = 0; i < n; i++) {
    nbd[i] = 1; // no upper bound, just lower
    l[i] = 0.;
    u[i] = 0.;
  }
  // Setup initial condition, get from W
  copy_mat_2_vec(W,x);

  /*     We start the iteration by initializing task. */
  *task = (integer)START;
  int iter=0;
  while ((iter < maxiter) && !checkpoint_and_exit) {
    setulb(&n, &m, x, l, u, nbd, &f, g, &factr, &pgtol, wa, iwa, task, 
           &iprint, csave, lsave, isave, dsave);
    if (IS_FG(*task)) {
      // update cost and gradient
      copy_vec_2_mat(x,W);
      f=cost(W,X,Y,Lx,Ly,Omega,lambda);
      gradient(gmat,W,X,Y,Lx,Ly,Omega,lambda,YXT,XXT,LxLxT,LyLyT);
      copy_mat_2_vec(gmat,g);
    } else if (*task==NEW_X) {
      // new iterate
      iter++;
    } else if (*task==ABNORMAL) {
      cout << "WARNING: ABNORMAL_TERMINATION_IN_LNSRCH" << endl;
    } else if (*task==RESTART) {
      cout << "WARNING: ABNORMAL_TERMINATION_IN_LNSRCH" << endl;
    } else if (IS_WARNING(*task)) {
      cout << "WARNING: " << *task << endl;
    } else if (IS_ERROR(*task)) { 
      cout << "ERROR: " << *task << endl;
    } else if (IS_STOP(*task)) {
      cout << "STOP: " << *task << endl;
      break;
    } else if (IS_CONVERGED(*task)) {
      cout << "CONVERGED!" << endl;
      converged=1;
      break;
    } else {
      cout << "Unknown status, task=" << *task << endl;
    }
    if ( (checkpt_file != NULL) && (checkpt_iter > 0) &&
         ((iter % checkpt_iter) == 0) && (iter != 0) ) {
      // checkpoint iterate
      copy_vec_2_mat(x,W);
      if ( !save_matrix(checkpt_file, W) ) {
        cout << "Error writing checkpoint file" << endl;
        converged=0;
        break;
      }
    }
  }
  copy_vec_2_mat(x,W);
  // free arrays
  free(x); free(g); free(u); free(l); free(nbd); free(wa); free(iwa);

  if (converged) {
    retcode=0;
  } else if (checkpoint_and_exit) {
    cout << "Caught TERM signal, checkpointing then exiting" << endl;
    retcode=1;
  } else if (iter == maxiter) {
    cout << "Maximum number of iterations reached" << endl;
    retcode=2;
  } else if (!converged) {
    retcode=3; // catchall for other errors
  }
  return(retcode);
}

