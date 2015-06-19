#include "functions.hpp"


double loss(mat W, mat X, mat Y) {
  /* || Y - W*X ||_F^2  */
  return pow(norm(Y-W*X,"fro"),2);
}

double regularizer(mat W, sp_mat Lx, sp_mat Ly) {
  /* || W*Lx' + Ly*W ||_F^2 */
  return pow(norm(W*Lx.t() + Ly*W,"fro"),2);
}

double cost(mat W, mat X, mat Y,
            sp_mat Lx, sp_mat Ly, double lambda) {
  /* Returns the cost of the current parameters */
  double cost1, cost2;
  cost1 = loss(W,X,Y);
  cost2 = regularizer(W,Lx,Ly);
  return cost1+lambda*lambda*cost2;
}

mat gradient(mat W, sp_mat Lx, sp_mat Ly, double lambda,
             mat YXT, mat XXT, 
             sp_mat LxLxT, sp_mat LyLyT) {
  /* Computes the gradient g, which is modified in-place */
  return -2*YXT + 2*W*XXT + 2*lambda*(2*Ly*W*Lx + W*LxLxT + LyLyT*W);
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

void copy_mat_2_vec(mat A, double *v) {
  long m=(long) A.n_rows;
  long n=(long) A.n_cols;
  long i,j;
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v[i*n+j]=A(i,j); // could cause buffer overflow if v is wrong size
    }
  }
}

sp_mat arma_sp_mat_mmread(char *fn) {
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
      exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0) {
      printf("Could not process Matrix Market banner.\n");
      exit(1);
    }

    if (!(mm_is_matrix(matcode)) || mm_is_dense(matcode)
        || mm_is_complex(matcode)) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nz)) !=0) {
      printf("Error reading matrix header\n");
      exit(1);
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
      sp_mat M=Ml+Ml.t();
      M.diag() /= 2;
      return M;
    } else {
      sp_mat M(locations,values,m,n);
      return M;
    }
}

mat arma_mat_mmread(char *fn) {
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
      exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0) {
      printf("Could not process Matrix Market banner.\n");
      exit(1);
    }

    if (!(mm_is_matrix(matcode)) || mm_is_sparse(matcode) || 
        mm_is_complex(matcode)) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of matrix .... */
    if ((ret_code = mm_read_mtx_array_size(f, &m, &n)) !=0) {
      printf("Error reading matrix header\n");
      exit(1);
    }

    mat M(m, n);        

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
    return M;
}

int arma_mat_mmwrite(char *fn, mat M) {
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
                  const sp_mat &Lx, const sp_mat &Ly, double lambda,
                  int maxiter, double factr, double pgtol) {
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
  if (lambda <= 0.0) {
    printf("lambda should be > 0\n");
    return(1);
  }

  // Do a few computations to setup matrix products for gradient
  mat YXT,XXT;
  sp_mat LxLxT,LyLyT;
  YXT=Y*X.t();
  XXT=X*X.t();
  LxLxT=Lx*Lx.t();
  LyLyT=Ly*Ly.t();
  // Initial test
  double f;
  mat gmat;
  f=cost(W,X,Y,Lx,Ly,lambda);
  cout << "initial cost: " << f << endl;
  gmat=gradient(W,Lx,Ly,lambda,YXT,XXT,LxLxT,LyLyT);
  cout << "initial gradient norm: " << norm(gmat,"fro") << endl;
  
  // Run L-BFGS-B optimization
  /* Local variables */
  double *g;
  integer i;
  double *l,*u;
  integer m, n;
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
  m = 5;
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
  while(iter < maxiter) {
    setulb(&n, &m, x, l, u, nbd, &f, g, &factr, &pgtol, wa, iwa, task, 
           &iprint, csave, lsave, isave, dsave);
    if (IS_FG(*task)) {
      // update cost and gradient (in-place)
      copy_vec_2_mat(x,W);
      f=cost(W,X,Y,Lx,Ly,lambda);
      gmat=gradient(W,Lx,Ly,lambda,YXT,XXT,LxLxT,LyLyT);
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
      break;
    } else {
      cout << "Unknown status, task=" << *task << endl;
    }
  }
  copy_vec_2_mat(x,W);
  // free arrays
  free(x); free(g); free(u); free(l); free(nbd); free(wa); free(iwa);

  if (iter == maxiter) {
    cout << "Maximum number of iterations reached" << endl;
    return(2);
  } else {
    return(0);
  }
}
