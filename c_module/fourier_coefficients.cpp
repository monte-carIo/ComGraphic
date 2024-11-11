#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>        // For std::unordered_map and other STL types
#include <pybind11/complex.h>    // For std::complex type
#include <complex>
#include <unordered_map>
#include <cstddef>               // For std::ptrdiff_t

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;
using namespace std;

// Define custom hash function for std::pair<int, int>
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>()(p.first) ^ (std::hash<T2>()(p.second) << 1);
    }
};

unordered_map<pair<int, int>, complex<double>, pair_hash> 
calculate_coefficients(py::array_t<double> Z, py::array_t<double> X, py::array_t<double> Y, double dx, double dy, double area, int L) {
    
    auto Z_buf = Z.unchecked<2>(); 
    auto X_buf = X.unchecked<2>();
    auto Y_buf = Y.unchecked<2>();

    unordered_map<pair<int, int>, complex<double>, pair_hash> coefficients;
    
    for (int m = -100; m <= 100; ++m) {
        for (int n = -100; n <= 100; ++n) {
            complex<double> C_mn = 0.0;

            for (std::ptrdiff_t i = 0; i < Z_buf.shape(0); ++i) { // Changed ssize_t to std::ptrdiff_t
                for (std::ptrdiff_t j = 0; j < Z_buf.shape(1); ++j) { // Changed ssize_t to std::ptrdiff_t
                    double exp_factor = -2.0 * M_PI * (m * X_buf(i, j) + n * Y_buf(i, j)) / (2 * L);
                    complex<double> exp_val = polar(1.0, exp_factor);
                    C_mn += Z_buf(i, j) * exp_val * dx * dy;
                }
            }

            C_mn /= area;
            coefficients[{m, n}] = C_mn;
        }
    }

    return coefficients;
}

PYBIND11_MODULE(fourier_module, m) {
    m.def("calculate_coefficients", &calculate_coefficients, "Calculate Fourier coefficients",
          py::arg("Z"), py::arg("X"), py::arg("Y"), py::arg("dx"), py::arg("dy"), py::arg("area"), py::arg("L"));
}
