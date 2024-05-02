//#include <iostream>
//#include "cmath"
//#include "vector"
//#include "utility"
//
//class Vector2d {
//public:
//    double data[2];
//
//    Vector2d(double x = 0, double y = 0) {
//        data[0] = x;
//        data[1] = y;
//    }
//
//
//    // overload [] for accessing element
//
//    // returns mutable refernce
//    double &operator[](int index) {
//        return data[index];
//    }
//
//    // immutable reference
//    const double &operator[](int index) const {
//        return data[index];
//    }
//
//    // function for calculating Euclidian norm od hte vector
//    double norm() const {
//        return std::sqrt(data[0] * data[0] + data[1] * data[1]);
//    }
//
//    // overload += for adding vector
//    Vector2d &operator+=(const Vector2d &other) {
//        data[0] += other.data[0];
//        data[1] += other.data[1];
//        return *this;
//    }
//
//    // overload - operation to work for vector
//    Vector2d operator-(const Vector2d &other) const {
//        return Vector2d(data[0] - other[0], data[1] - data[1]);
//    }
//
//    // overload * for multiplying vector by scalar
//    Vector2d operator*(double scalar) const {
//        return Vector2d(data[0] * scalar, data[1] * scalar);
//    }
//};
//
//// Define class to represent 2*2 matrix of doubles
//class Matrix2d {
//public:
//    double data[2][2];
//
//    Matrix2d(double a11 = 0, double a12 = 0, double a21 = 0, double a22 = 0) {
//        data[0][0] = a11;
//        data[0][1] = a12;
//        data[1][0] = a21;
//        data[1][1] = a22;
//    }
//
//    // overload operator for accessnig element
//
//    // returns mutable pointer to the element at the give nindex
//    double *operator[](int index) {
//        return data[index];
//    }
//
//    // returns immutable pointer
//    const double *operator[](int index) const {
//        return data[index];
//    }
//
//
//    // calculate inverse of a matrix using Gaussian elimination
//    Matrix2d inverse() const {
//        // copy the matrix
//        Matrix2d inv(data[0][0], data[0][1], data[1][0], data[1][1]);
//
//        // create identity matrix
//        Matrix2d identity(1, 0, 0, 1);
//
//        // pivoting
//        for (int i = 0; i < 2; i++) {
//            int maxRow = i;
//            for (int j = i + 1; j < 2; j++) {
//                if (std::fabs(inv[j][i]) > std::fabs(inv[maxRow][i])) {
//                    maxRow = j;
//                }
//            }
//
//            // swap rows if neccary
//
//            if (maxRow != i) {
//                for (int k = 0; k < 2; k++) {
//                    std::swap(inv[i][k], inv[maxRow][k]);
//                    std::swap(identity[i][k], identity[maxRow][k]);
//                }
//            }
//
//            // perform elimination
//
//            for (int j = i + 1; j < 2; j++) {
//                double factor = inv[j][i] / inv[i][i];
//                for (int k = i; k < 2; k++) {
//                    inv[j][k] -= factor * inv[i][k];
//                }
//                for (int k = 0; k < 2; k++) {
//                    identity[j][k] -= factor * identity[i][k];
//                }
//            }
//        }
//
//        // Back substitution
//        for (int i = 1; i >= 0; i--) {
//            double divisor = inv[i][i];
//            for (int k = 0; k < 2; k++) {
//                inv[i][k] /= divisor;
//                identity[i][k] /= divisor;
//            }
//            for (int j = i - 1; j >= 0; j--) {
//                double factor = inv[j][i];
//                for (int k = 0; k < 2; k++) {
//                    inv[j][k] -= factor * inv[i][k];
//                    identity[j][k] -= factor * identity[i][k];
//                }
//            }
//        }
//        return identity;
//    }
//
//    // overload * to multiply the matrix by a vector
//
//    Vector2d operator*(const Vector2d &v) const {
//        return Vector2d(
//                data[0][0] * v[0] + data[0][1] * v[1],
//                data[1][0] * v[0] + data[1][1] * v[1]
//        );
//    }
//};
//
//// define Rosenbrock function
//
//double rosenbrock(const Vector2d &x) {
//    return 100.0 * std::pow(x[1] - std::pow(x[0], 2.0), 2.0) + std::pow(1 - x[0], 2.0);
//}
//
//// Gradient of Rosenbrock function
//Vector2d rosenbrock_grad(const Vector2d &x) {
//    return Vector2d(
//            -400.0 * x[0] * (x[1] - std::pow(x[0], 2.0)) - 2.0 * (1 - x[0]),
//            200.0 * (x[1] - std::pow(x[0], 2.0))
//    );
//}
//
//// Hessian matrix of Rosenbrock function
//Matrix2d rosenbrock_hess(const Vector2d &x) {
//    return Matrix2d(
//            1200.0 * std::pow(x[0], 2.0) - 400.0 * x[1] + 2,
//            -400.0 * x[0],
//            -400.0 * x[0],
//            200.0
//    );
//}
//
//
//// implement Newton method for optimization
//
//std::pair<Vector2d, int> newtons_method(const Vector2d &start_point, double tol = 1e-5, int max_iter = 1000) {
//    Vector2d x = start_point;
//    for (int i = 0; i < max_iter; i++) {
//        Vector2d grad = rosenbrock_grad(x);
//        Matrix2d hess = rosenbrock_hess(x);
//
//        // If the norm of the gradient is less than the tolerance, stop
//        if (grad.norm() < tol) {
//            return std::make_pair(x, i);
//        }
//
//        // Calculate the step using the inverse of the Hessian and the gradient
//        Matrix2d inv_hess = hess.inverse();
//        Vector2d delta = inv_hess * (-grad);
//
//        // Update x
//        x += delta;
//    }
//    return std::make_pair(x, max_iter);
//}
//
//
////  test with different starting points
//int main() {
//    std::vector<Vector2d> starting_points = {
//            Vector2d(-1, -1),
//            Vector2d(0, 0),
//            Vector2d(2, 2),
//            Vector2d(1.5, 1.5)
//    };
//
//    for (const auto &point : starting_points) {
//        auto [solution, iterations] = newtons_method(point);
//        std::cout << "Starting point: (" << point.data[0] << ", " << point.data[1] << ")"
//                  << ", Solution: (" << solution.data[0] << ", " << solution.data[1] << ")"
//                  << ", Iterations: " << iterations << std::endl;
//    }
//
//    return 0;
//}
//
//
//
//
//
//
//
//


#include <iostream>
#include <cmath>
#include <vector>
#include <utility>

// Class representing a 2D vector
class Vector2d {
public:
    double data[2];

    // Constructor to initialize the vector
    Vector2d(double x = 0, double y = 0) {
        data[0] = x;
        data[1] = y;
    }

    // Overload [] for accessing elements (mutable and immutable references)
    double &operator[](int index) {
        return data[index];
    }

    const double &operator[](int index) const {
        return data[index];
    }

    // Calculate the Euclidean norm of the vector
    double norm() const {
        return std::sqrt(data[0] * data[0] + data[1] * data[1]);
    }

    // Overload += for adding another vector
    Vector2d &operator+=(const Vector2d &other) {
        data[0] += other.data[0];
        data[1] += other.data[1];
        return *this;
    }

    // Overload - for subtracting another vector
    Vector2d operator-(const Vector2d &other) const {
        return Vector2d(data[0] - other.data[0], data[1] - other.data[1]);
    }

    // Overload * for multiplying the vector by a scalar
    Vector2d operator*(double scalar) const {
        return Vector2d(data[0] * scalar, data[1] * scalar);
    }

    // Overload - unary operator for negating the vector
    Vector2d operator-() const {
        return Vector2d(-data[0], -data[1]);
    }
};

// Class representing a 2x2 matrix
class Matrix2d {
public:
    double data[2][2];

    // Constructor to initialize the matrix
    Matrix2d(double a11 = 0, double a12 = 0, double a21 = 0, double a22 = 0) {
        data[0][0] = a11;
        data[0][1] = a12;
        data[1][0] = a21;
        data[1][1] = a22;
    }

    // Overload operator[] for accessing elements (mutable and immutable pointers)
    double* operator[](int index) {
        return data[index];
    }

    const double* operator[](int index) const {
        return data[index];
    }

    // Calculate the inverse of the matrix using Gaussian elimination
    Matrix2d inverse() const {
        // Create a copy of the matrix
        Matrix2d inv(data[0][0], data[0][1], data[1][0], data[1][1]);

        // Create the identity matrix
        Matrix2d identity(1, 0, 0, 1);

        // Perform Gaussian elimination and back substitution
        for (int i = 0; i < 2; i++) {
            // Find the row with the largest absolute value in the current column for pivoting
            int maxRow = i;
            for (int j = i + 1; j < 2; j++) {
                if (std::fabs(inv[j][i]) > std::fabs(inv[maxRow][i])) {
                    maxRow = j;
                }
            }

            // Swap rows in both inv and identity if necessary
            if (maxRow != i) {
                for (int k = 0; k < 2; k++) {
                    std::swap(inv[i][k], inv[maxRow][k]);
                    std::swap(identity[i][k], identity[maxRow][k]);
                }
            }

            // Eliminate values below the pivot
            for (int j = i + 1; j < 2; j++) {
                double factor = inv[j][i] / inv[i][i];
                for (int k = i; k < 2; k++) {
                    inv[j][k] -= factor * inv[i][k];
                }
                for (int k = 0; k < 2; k++) {
                    identity[j][k] -= factor * identity[i][k];
                }
            }
        }

        // Back substitution to obtain the inverse
        for (int i = 1; i >= 0; i--) {
            double divisor = inv[i][i];
            for (int k = 0; k < 2; k++) {
                inv[i][k] /= divisor;
                identity[i][k] /= divisor;
            }
            for (int j = i - 1; j >= 0; j--) {
                double factor = inv[j][i];
                for (int k = 0; k < 2; k++) {
                    inv[j][k] -= factor * inv[i][k];
                    identity[j][k] -= factor * identity[i][k];
                }
            }
        }

        return identity;
    }

    // Overload * operator to multiply the matrix by a vector
    Vector2d operator*(const Vector2d& v) const {
        return Vector2d(
                data[0][0] * v[0] + data[0][1] * v[1],
                data[1][0] * v[0] + data[1][1] * v[1]
        );
    }
};

// Define the Rosenbrock function
double rosenbrock(const Vector2d &x) {
    return 100.0 * std::pow(x[1] - std::pow(x[0], 2.0), 2.0) + std::pow(1 - x[0], 2.0);
}

// Gradient of the Rosenbrock function
Vector2d rosenbrock_grad(const Vector2d &x) {
    return Vector2d(
            -400.0 * x[0] * (x[1] - std::pow(x[0], 2.0)) - 2.0 * (1 - x[0]),
            200.0 * (x[1] - std::pow(x[0], 2.0))
    );
}

// Hessian matrix of the Rosenbrock function
Matrix2d rosenbrock_hess(const Vector2d &x) {
    return Matrix2d(
            1200.0 * std::pow(x[0], 2.0) - 400.0 * x[1] + 2,
            -400.0 * x[0],
            -400.0 * x[0],
            200.0
    );
}

// Implement Newton's method for optimization
std::pair<Vector2d, int> newtons_method(const Vector2d &start_point, double tol = 1e-5, int max_iter = 1000) {
    Vector2d x = start_point;
    for (int i = 0; i < max_iter; i++) {
        Vector2d grad = rosenbrock_grad(x);
        Matrix2d hess = rosenbrock_hess(x);

        // If the norm of the gradient is less than the tolerance, stop
        if (grad.norm() < tol) {
            return std::make_pair(x, i);
        }

        // Calculate the step using the inverse of the Hessian and the gradient
        Matrix2d inv_hess = hess.inverse();
        Vector2d delta = inv_hess * (-grad);

        // Update x
        x += delta;
    }
    return std::make_pair(x, max_iter);
}

// Test with different starting points
int main() {
    std::vector<Vector2d> starting_points = {
            Vector2d(-1, -1),
            Vector2d(0, 0),
            Vector2d(2, 2),
            Vector2d(1.5, 1.5)
    };

    for (const auto &point : starting_points) {
        auto [solution, iterations] = newtons_method(point);
        std::cout << std::fixed;
        std::cout << "Starting point: (" << point[0] << ", " << point[1] << ")"
                  << ", Solution: (" << solution[0] << ", " << solution[1] << ")"
                  << ", Iterations: " << iterations << std::endl;
    }

    return 0;
}
// output
//Starting point: (-1.000000, -1.000000), Solution: (1.000000, 1.000000), Iterations: 5
//Starting point: (0.000000, 0.000000), Solution: (1.000000, 1.000000), Iterations: 2
//Starting point: (2.000000, 2.000000), Solution: (1.000000, 1.000000), Iterations: 5
//Starting point: (1.500000, 1.500000), Solution: (1.000000, 1.000000), Iterations: 5

// Regardless of the starting point ((-1, -1), (0, 0), (2, 2), or (1.5, 1.5)), the method converges to the same solution: (1, 1)

// The method converges in a few iterations (2, 5, or 5 iterations) depending on the starting point