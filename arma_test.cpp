#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char** argv) {
  arma_rng::set_seed_random();
  mat A = randu<mat>(4,4);
  sp_mat B = sprandu<sp_mat>(4,4,0.2);
  
  cout << A << endl;
  cout << B << endl;

  cout << A*B.t() << endl;
  
  return 0;
}
