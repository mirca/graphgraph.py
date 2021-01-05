#include "operators.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;


PYBIND11_MODULE(operators, m) {
  m.doc() = "laplacian, adjacency, and degree operators and their adjoints";

  m.def("laplacian_op", &laplacian_op,
        py::arg("weights"),
        R"pbdoc(
          Computes the Laplacian linear operator which maps a vector of weights into
          a valid Laplacian matrix.

          param: weights vector of graph weights
          return: Laplacian matrix
         )pbdoc"
        );

  m.def("inv_laplacian_op", &inv_laplacian_op,
        py::arg("laplacian_matrix"),
        R"pbdoc(
          Computes the inverse mapping of the Laplacian linear operator which
          maps a Laplacian matrix into a valid vector of weights

          param: laplacian_matrix laplacian matrix
          return: weights vector of graph weights
         )pbdoc"
        );

  m.def("adjacency_op", &adjacency_op,
        py::arg("weights"),
        R"pbdoc(
          Computes the Adjacency linear operator which maps a vector of weights into
          a valid adjacency matrix.

          param: weights vector of graph weights
          return: adjacency matrix
         )pbdoc"
        );

  m.def("inv_adjacency_op", &inv_adjacency_op,
        py::arg("adjacency_matrix"),
        R"pbdoc(
          Computes the inverse mapping of the Adjacency linear operator which
          maps an adjacency matrix into a valid vector of weights

          param: adjacency_matrix adjacency matrix
          return: weights vector of graph weights
         )pbdoc"
        );

  m.def("degree_op", &degree_op,
        py::arg("weights"),
        R"pbdoc(
          Computes the Degree linear operator which maps a vector of weights into
          a valid degree vector.

          param: weights vector of graph weights
          return: vector of degrees
         )pbdoc"
        );

  m.def("adj_laplacian_op", &adj_laplacian_op,
        py::arg("in_matrix"),
        R"pbdoc(
          Computes the adjoint of the Laplacian operator.

          param: in_matrix input matrix
          return: out_vector output vector
         )pbdoc"
        );

  m.def("adj_adjacency_op", &adj_adjacency_op,
        py::arg("in_matrix"),
        R"pbdoc(
          Computes the adjoint of the adjacency operator.

          param: in_matrix input matrix
          return: out_vector output vector
         )pbdoc"
        );

  m.def("adj_degree_op", &adj_degree_op,
        py::arg("weights"),
        R"pbdoc(
          Computes the adjoint of the degree operator.

          param: in_vector input vector
          return: out_vector output vector
         )pbdoc"
        );

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
