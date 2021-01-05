/*  Please compile with 
 *  g++ -I/usr/local/Cellar/eigen/3.3.7/include/eigen3/ -O2 -DNDEBUG test_eigen.cpp -o test_eigen 
*   Make sure use -O2 for optimization.
*/

#include <iostream>
#include "Eigen/Dense"
using namespace Eigen;
int main()
{

  int n_a_rows = 8000;
  int n_a_cols = 6000;
  int n_b_rows = n_a_cols;
  int n_b_cols = 4000;

  MatrixXd a(n_a_rows, n_a_cols);

  for (int i = 0; i < n_a_rows; ++ i)
      for (int j = 0; j < n_a_cols; ++ j)
        a (i, j) = n_a_cols * i + j+0.0;

  MatrixXd b (n_b_rows, n_b_cols);
  for (int i = 0; i < n_b_rows; ++ i)
      for (int j = 0; j < n_b_cols; ++ j)
        b (i, j) = n_b_cols * i + j+0.0;

  MatrixXi d (n_a_rows, n_b_cols);

  clock_t begin = clock();

  d = a * b;

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  std::cout << "Time taken : " << elapsed_secs << std::endl;

}
