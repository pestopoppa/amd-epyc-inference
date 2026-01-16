/**
 * @file bayesopt.cpp
 * @brief Bayesian Optimization with Gaussian Process surrogate
 *
 * Maximizes an objective function using GP-based acquisition functions.
 *
 * Usage:
 *   {"command": "bayesopt", "bounds": [[0,5], [0,5]], "objective": "-(x0-2)**2-(x1-3)**2", "n_iter": 20}
 *
 * Response:
 *   {"status": "success", "result": {"x_best": [2.0, 3.0], "y_best": 0.0, ...}}
 */

#include "../include/command.hpp"
#include "../include/expression.hpp"
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

class BayesOptCommand : public Command {
public:
    std::string name() const override { return "bayesopt"; }
    std::string description() const override {
        return "Bayesian optimization with Gaussian process";
    }

    json execute(const json& params) override {
        Timer timer;

        try {
            // Parse parameters
            if (!params.contains("bounds")) {
                return error("Missing required parameter: bounds");
            }
            if (!params.contains("objective")) {
                return error("Missing required parameter: objective");
            }

            auto bounds_json = params["bounds"];
            std::string objective_expr = params["objective"];
            int n_init = params.value("n_init", 5);
            int n_iter = params.value("n_iter", 25);
            std::string acquisition = params.value("acquisition", "ei");
            double noise = params.value("noise", 1e-6);
            uint64_t seed = params.value("seed", 42);

            // Parse bounds
            size_t dim = bounds_json.size();
            std::vector<std::pair<double, double>> bounds;
            for (const auto& b : bounds_json) {
                bounds.push_back({b[0].get<double>(), b[1].get<double>()});
            }

            // Initialize RNG
            std::mt19937_64 rng(seed);

            // Expression parser for objective
            Expression objective(objective_expr);

            // Storage for observations
            std::vector<std::vector<double>> X;
            std::vector<double> y;

            // Initial random samples (Latin Hypercube-ish)
            for (int i = 0; i < n_init; ++i) {
                std::vector<double> x(dim);
                for (size_t d = 0; d < dim; ++d) {
                    std::uniform_real_distribution<double> dist(bounds[d].first, bounds[d].second);
                    x[d] = dist(rng);
                }
                X.push_back(x);
                y.push_back(objective.evaluate(x));
            }

            // GP hyperparameters (simple defaults)
            double length_scale = 1.0;
            double signal_var = 1.0;

            // Best found so far
            auto best_idx = std::max_element(y.begin(), y.end()) - y.begin();
            std::vector<double> x_best = X[best_idx];
            double y_best = y[best_idx];

            // History for output
            json history_X = json::array();
            json history_y = json::array();
            for (size_t i = 0; i < X.size(); ++i) {
                history_X.push_back(X[i]);
                history_y.push_back(y[i]);
            }

            // Bayesian optimization loop
            for (int iter = 0; iter < n_iter; ++iter) {
                size_t n = X.size();

                // Build GP kernel matrix K
                Eigen::MatrixXd K(n, n);
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        K(i, j) = rbf_kernel(X[i], X[j], length_scale, signal_var);
                    }
                    K(i, i) += noise;  // Add noise to diagonal
                }

                // Compute K inverse (using Cholesky)
                Eigen::LLT<Eigen::MatrixXd> llt(K);
                if (llt.info() != Eigen::Success) {
                    // Regularize more
                    K.diagonal().array() += 1e-4;
                    llt.compute(K);
                }

                Eigen::VectorXd y_vec(n);
                for (size_t i = 0; i < n; ++i) {
                    y_vec(i) = y[i];
                }

                Eigen::VectorXd alpha = llt.solve(y_vec);

                // Find next point by maximizing acquisition function
                // Use random search for simplicity
                std::vector<double> x_next(dim);
                double best_acq = -std::numeric_limits<double>::infinity();

                int n_candidates = 1000;
                for (int c = 0; c < n_candidates; ++c) {
                    std::vector<double> x_cand(dim);
                    for (size_t d = 0; d < dim; ++d) {
                        std::uniform_real_distribution<double> dist(bounds[d].first, bounds[d].second);
                        x_cand[d] = dist(rng);
                    }

                    // Compute GP prediction at candidate
                    Eigen::VectorXd k_star(n);
                    for (size_t i = 0; i < n; ++i) {
                        k_star(i) = rbf_kernel(x_cand, X[i], length_scale, signal_var);
                    }

                    double mu = k_star.dot(alpha);
                    double k_ss = rbf_kernel(x_cand, x_cand, length_scale, signal_var);
                    Eigen::VectorXd v = llt.matrixL().solve(k_star);
                    double sigma2 = std::max(0.0, k_ss - v.dot(v));
                    double sigma = std::sqrt(sigma2);

                    // Compute acquisition value
                    double acq = compute_acquisition(mu, sigma, y_best, acquisition);

                    if (acq > best_acq) {
                        best_acq = acq;
                        x_next = x_cand;
                    }
                }

                // Evaluate objective at new point
                double y_next = objective.evaluate(x_next);
                X.push_back(x_next);
                y.push_back(y_next);

                history_X.push_back(x_next);
                history_y.push_back(y_next);

                // Update best
                if (y_next > y_best) {
                    x_best = x_next;
                    y_best = y_next;
                }

                // Update length scale (simple adaptation)
                if ((iter + 1) % 10 == 0) {
                    length_scale = estimate_length_scale(X, bounds);
                }
            }

            // Build result
            json result;
            result["x_best"] = x_best;
            result["y_best"] = y_best;
            result["history"]["X"] = history_X;
            result["history"]["y"] = history_y;
            result["n_evaluations"] = static_cast<int>(X.size());
            result["model"]["length_scale"] = length_scale;
            result["model"]["signal_variance"] = signal_var;
            result["model"]["noise"] = noise;

            json stats;
            stats["elapsed_ms"] = timer.elapsed_ms();
            stats["n_init"] = n_init;
            stats["n_iter"] = n_iter;
            stats["acquisition"] = acquisition;

            return success(result, stats);

        } catch (const std::exception& e) {
            return error(std::string("BayesOpt error: ") + e.what());
        }
    }

private:
    /**
     * @brief RBF (Squared Exponential) kernel
     */
    double rbf_kernel(const std::vector<double>& x1, const std::vector<double>& x2,
                      double length_scale, double signal_var) {
        double sq_dist = 0.0;
        for (size_t i = 0; i < x1.size(); ++i) {
            double diff = x1[i] - x2[i];
            sq_dist += diff * diff;
        }
        return signal_var * std::exp(-0.5 * sq_dist / (length_scale * length_scale));
    }

    /**
     * @brief Compute acquisition function value
     */
    double compute_acquisition(double mu, double sigma, double y_best,
                               const std::string& acq_type) {
        if (sigma < 1e-10) {
            return mu > y_best ? 1e10 : -1e10;
        }

        double z = (mu - y_best) / sigma;

        if (acq_type == "ei") {
            // Expected Improvement
            double pdf = std::exp(-0.5 * z * z) / std::sqrt(2 * M_PI);
            double cdf = 0.5 * (1 + std::erf(z / std::sqrt(2)));
            return sigma * (z * cdf + pdf);
        } else if (acq_type == "ucb") {
            // Upper Confidence Bound
            double beta = 2.0;  // Exploration parameter
            return mu + beta * sigma;
        } else if (acq_type == "pi") {
            // Probability of Improvement
            return 0.5 * (1 + std::erf(z / std::sqrt(2)));
        }

        return mu;  // Default to mean
    }

    /**
     * @brief Estimate length scale from data
     */
    double estimate_length_scale(const std::vector<std::vector<double>>& X,
                                  const std::vector<std::pair<double, double>>& bounds) {
        // Use median pairwise distance
        std::vector<double> distances;
        for (size_t i = 0; i < X.size(); ++i) {
            for (size_t j = i + 1; j < X.size(); ++j) {
                double dist = 0.0;
                for (size_t d = 0; d < X[i].size(); ++d) {
                    double range = bounds[d].second - bounds[d].first;
                    double diff = (X[i][d] - X[j][d]) / range;
                    dist += diff * diff;
                }
                distances.push_back(std::sqrt(dist));
            }
        }

        if (distances.empty()) return 1.0;

        std::sort(distances.begin(), distances.end());
        return distances[distances.size() / 2];
    }
};

// Factory function
std::unique_ptr<Command> create_bayesopt() {
    return std::make_unique<BayesOptCommand>();
}
