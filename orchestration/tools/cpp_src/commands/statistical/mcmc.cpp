/**
 * @file mcmc.cpp
 * @brief Metropolis-Hastings MCMC sampler
 *
 * Samples from arbitrary distributions specified by log-density expressions.
 *
 * Usage:
 *   {"command": "mcmc", "log_density": "-0.5*(x0**2 + x1**2)", "x0": [0, 0], "n_samples": 5000}
 *
 * Response:
 *   {"status": "success", "result": {"samples": [...], "acceptance_rate": 0.234, ...}}
 */

#include "../include/command.hpp"
#include "../include/expression.hpp"
#include <random>
#include <vector>
#include <cmath>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

class MCMCCommand : public Command {
public:
    std::string name() const override { return "mcmc"; }
    std::string description() const override {
        return "Metropolis-Hastings MCMC sampler";
    }

    json execute(const json& params) override {
        Timer timer;

        try {
            // Parse parameters
            if (!params.contains("log_density")) {
                return error("Missing required parameter: log_density");
            }
            if (!params.contains("x0")) {
                return error("Missing required parameter: x0");
            }

            std::string log_density_expr = params["log_density"];
            std::vector<double> x0 = params["x0"].get<std::vector<double>>();
            int n_samples = params.value("n_samples", 10000);
            double proposal_std = params.value("proposal_std", 1.0);
            int burnin = params.value("burnin", 1000);
            int thin = params.value("thin", 1);
            uint64_t seed = params.value("seed", 42);

            size_t dim = x0.size();
            if (dim == 0) {
                return error("x0 must have at least one element");
            }

            // Initialize RNG
            std::mt19937_64 rng(seed);
            std::normal_distribution<double> proposal(0.0, proposal_std);
            std::uniform_real_distribution<double> uniform(0.0, 1.0);

            // Expression parser
            Expression log_density(log_density_expr);

            // Current state
            std::vector<double> x_current = x0;
            double log_p_current = log_density.evaluate(x_current);

            // Storage for samples
            int total_iterations = burnin + n_samples * thin;
            std::vector<std::vector<double>> samples;
            samples.reserve(n_samples);

            // Statistics
            int accepted = 0;
            int total_proposals = 0;

            // MCMC loop
            for (int i = 0; i < total_iterations; ++i) {
                // Propose new state
                std::vector<double> x_proposed = x_current;
                for (size_t d = 0; d < dim; ++d) {
                    x_proposed[d] += proposal(rng);
                }

                // Evaluate log density at proposed state
                double log_p_proposed = log_density.evaluate(x_proposed);

                // Accept/reject
                double log_alpha = log_p_proposed - log_p_current;
                total_proposals++;

                if (log_alpha >= 0 || std::log(uniform(rng)) < log_alpha) {
                    x_current = x_proposed;
                    log_p_current = log_p_proposed;
                    accepted++;
                }

                // Store sample (after burnin, with thinning)
                if (i >= burnin && (i - burnin) % thin == 0) {
                    samples.push_back(x_current);
                }
            }

            // Compute statistics
            double acceptance_rate = static_cast<double>(accepted) / total_proposals;

            // Mean
            std::vector<double> mean(dim, 0.0);
            for (const auto& sample : samples) {
                for (size_t d = 0; d < dim; ++d) {
                    mean[d] += sample[d];
                }
            }
            for (size_t d = 0; d < dim; ++d) {
                mean[d] /= samples.size();
            }

            // Covariance
            std::vector<std::vector<double>> covariance(dim, std::vector<double>(dim, 0.0));
            for (const auto& sample : samples) {
                for (size_t i = 0; i < dim; ++i) {
                    for (size_t j = 0; j < dim; ++j) {
                        covariance[i][j] += (sample[i] - mean[i]) * (sample[j] - mean[j]);
                    }
                }
            }
            for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    covariance[i][j] /= (samples.size() - 1);
                }
            }

            // Standard deviation
            std::vector<double> std_dev(dim);
            for (size_t d = 0; d < dim; ++d) {
                std_dev[d] = std::sqrt(covariance[d][d]);
            }

            // Effective sample size (using autocorrelation at lag 1)
            std::vector<double> ess(dim);
            for (size_t d = 0; d < dim; ++d) {
                double autocorr = 0.0;
                for (size_t i = 0; i < samples.size() - 1; ++i) {
                    autocorr += (samples[i][d] - mean[d]) * (samples[i + 1][d] - mean[d]);
                }
                autocorr /= ((samples.size() - 1) * covariance[d][d]);
                // Approximate ESS
                ess[d] = samples.size() * (1.0 - autocorr) / (1.0 + autocorr);
                ess[d] = std::max(1.0, std::min(ess[d], static_cast<double>(samples.size())));
            }

            // Build result
            json result;
            result["samples"] = samples;
            result["acceptance_rate"] = acceptance_rate;
            result["mean"] = mean;
            result["std_dev"] = std_dev;
            result["covariance"] = covariance;
            result["effective_sample_size"] = ess;
            result["n_samples"] = static_cast<int>(samples.size());
            result["dimension"] = static_cast<int>(dim);

            json stats;
            stats["elapsed_ms"] = timer.elapsed_ms();
            stats["total_iterations"] = total_iterations;
            stats["burnin"] = burnin;
            stats["thin"] = thin;

            return success(result, stats);

        } catch (const std::exception& e) {
            return error(std::string("MCMC error: ") + e.what());
        }
    }
};

// Factory function
std::unique_ptr<Command> create_mcmc() {
    return std::make_unique<MCMCCommand>();
}
