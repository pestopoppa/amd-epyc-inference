/**
 * @file integrate.cpp
 * @brief Numerical integration using adaptive Simpson's rule
 *
 * Implements adaptive quadrature for definite integrals.
 * No external dependencies beyond standard library.
 *
 * JSON Interface:
 * {
 *   "command": "integrate",
 *   "function": "sin(x)",        // Expression in x
 *   "a": 0.0,                    // Lower bound
 *   "b": 3.14159,                // Upper bound
 *   "tol": 1e-10,                // Tolerance (optional, default 1e-10)
 *   "max_depth": 50              // Max recursion depth (optional, default 50)
 * }
 *
 * Response:
 * {
 *   "status": "success",
 *   "result": 2.0,               // Integral value
 *   "stats": {
 *     "evaluations": 257,
 *     "elapsed_ms": 0.5
 *   }
 * }
 */

#include "../include/command.hpp"
#include "../include/expression.hpp"
#include <cmath>
#include <functional>

class IntegrateCommand : public Command {
public:
    std::string name() const override { return "integrate"; }
    std::string description() const override {
        return "Numerical integration using adaptive Simpson's rule";
    }

    json execute(const json& params) override {
        Timer timer;

        // Parse parameters
        if (!params.contains("function")) {
            return error("Missing 'function' parameter");
        }
        if (!params.contains("a") || !params.contains("b")) {
            return error("Missing 'a' or 'b' bound parameters");
        }

        std::string func_str = params["function"].get<std::string>();
        double a = params["a"].get<double>();
        double b = params["b"].get<double>();
        double tol = params.value("tol", 1e-10);
        int max_depth = params.value("max_depth", 50);

        // Create evaluator
        Expression expr(func_str);
        int eval_count = 0;

        auto f = [&](double x) -> double {
            eval_count++;
            return expr.evaluate({x});
        };

        try {
            double result = adaptive_simpson(f, a, b, tol, max_depth);

            json stats;
            stats["evaluations"] = eval_count;
            stats["elapsed_ms"] = timer.elapsed_ms();

            return success(result, stats);
        } catch (const std::exception& e) {
            return error(std::string("Integration failed: ") + e.what());
        }
    }

private:
    /**
     * @brief Adaptive Simpson's rule
     *
     * Uses Simpson's rule with adaptive subdivision for error control.
     */
    double adaptive_simpson(
        std::function<double(double)> f,
        double a, double b,
        double tol, int max_depth
    ) {
        // Initial Simpson's rule estimate
        double h = (b - a) / 2.0;
        double fa = f(a);
        double fb = f(b);
        double fm = f((a + b) / 2.0);
        double s = h * (fa + 4.0 * fm + fb) / 3.0;

        return adaptive_simpson_recursive(f, a, b, tol, s, fa, fb, fm, max_depth);
    }

    double adaptive_simpson_recursive(
        std::function<double(double)> f,
        double a, double b, double tol,
        double s, double fa, double fb, double fm,
        int depth
    ) {
        double c = (a + b) / 2.0;
        double h = (b - a) / 2.0;

        double flm = f((a + c) / 2.0);  // Left midpoint
        double frm = f((c + b) / 2.0);  // Right midpoint

        // Simpson's rule for left and right halves
        double sl = h * (fa + 4.0 * flm + fm) / 6.0;
        double sr = h * (fm + 4.0 * frm + fb) / 6.0;
        double s2 = sl + sr;

        // Richardson extrapolation error estimate
        double error = (s2 - s) / 15.0;

        if (depth <= 0 || std::abs(error) <= tol) {
            // Accept approximation with Richardson correction
            return s2 + error;
        }

        // Subdivide
        return adaptive_simpson_recursive(f, a, c, tol / 2.0, sl, fa, fm, flm, depth - 1)
             + adaptive_simpson_recursive(f, c, b, tol / 2.0, sr, fm, fb, frm, depth - 1);
    }
};

// Factory function
std::unique_ptr<Command> create_integrate() {
    return std::make_unique<IntegrateCommand>();
}
