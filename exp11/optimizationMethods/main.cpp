#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <string>
#include <sstream>
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

// Function for computing the gradient of the Rosenbrock function
std::vector<double> rosenbrock_grad(const std::vector<double>& x) {
    std::vector<double> grad(2);
    grad[0] = -400.0 * x[0] * (x[1] - pow(x[0], 2.0)) - 2.0 * (1 - x[0]);
    grad[1] = 200.0 * (x[1] - pow(x[0], 2.0));
    return grad;
}

// Steepest descent optimization function
std::pair<std::vector<double>, int> steepestDescent(const std::vector<double>& startPoint, double tol = 1e-5, int maxIter = 1000, double alpha = 1e-3) {
    std::vector<double> x = startPoint; // Copy startPoint

    for (int i = 0; i < maxIter; i++) {
        std::vector<double> grad = rosenbrock_grad(x);

        // Calculate the Euclidean norm of the gradient
        double gradNorm = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

        if (gradNorm < tol) {
            return {x, i}; // Return the result and iteration count
        }

        // Update x using the steepest descent update
        for (size_t j = 0; j < x.size(); j++) {
            x[j] -= alpha * grad[j];
        }
    }
    return {x, maxIter}; // Return max iterations if not converged
}

//// Test with different starting points the Newton method
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
//        std::cout << std::fixed;
//        std::cout << "Starting point: (" << point[0] << ", " << point[1] << ")"
//                  << ", Solution: (" << solution[0] << ", " << solution[1] << ")"
//                  << ", Iterations: " << iterations << std::endl;
//    }
//
//    return 0;
//}
//// output
////Starting point: (-1.000000, -1.000000), Solution: (1.000000, 1.000000), Iterations: 5
////Starting point: (0.000000, 0.000000), Solution: (1.000000, 1.000000), Iterations: 2
////Starting point: (2.000000, 2.000000), Solution: (1.000000, 1.000000), Iterations: 5
////Starting point: (1.500000, 1.500000), Solution: (1.000000, 1.000000), Iterations: 5
//
//// Regardless of the starting point ((-1, -1), (0, 0), (2, 2), or (1.5, 1.5)), the method converges to the same solution: (1, 1)
//
//// The method converges in a few iterations (2, 5, or 5 iterations) depending on the starting point



int main() {
    // Define the starting points
    std::vector<std::vector<double>> startingPoints = {
            {1.2, 1.2},
            {-1.2, 1.0}
    };

    std::vector<std::pair<std::vector<double>, int>> resultsSd;

    // Run the steepest descent optimization
    for (const auto& point : startingPoints) {
        resultsSd.push_back(steepestDescent(point));
    }

    // Display the results for steepest descent
    std::vector<std::string> resultsSdFormatted;
    for (size_t i = 0; i < startingPoints.size(); i++) {
        const auto& startPoint = startingPoints[i];
        const auto& result = resultsSd[i];
        const auto& solutionSd = result.first;
        int iterationsSd = result.second;

        // Format the results
        std::ostringstream resultStream;
        resultStream << "Starting point: [" << startPoint[0] << ", " << startPoint[1] << "], Solution: [" << solutionSd[0] << ", " << solutionSd[1] << "], Iterations: " << iterationsSd;

        resultsSdFormatted.push_back(resultStream.str());
    }

    // Print the formatted results
    for (const auto& result : resultsSdFormatted) {
        std::cout << result << std::endl;
    }

    return 0;
}

// result
//Starting point: [1.2, 1.2], Solution: [1.07889, 1.1643], Iterations: 1000
//Starting point: [-1.2, 1], Solution: [0.327263, 0.104013], Iterations: 1000

//while the steepest descent optimization seems to move the starting points closer to the global minimum,
// the process may need further optimization in terms of parameters or algorithm choice to improve performance
// and convergence speed.

