/**
 * @file symbolic_diff.cpp
 * @brief Symbolic differentiation using SymEngine
 *
 * DEPENDENCY: Requires SymEngine library
 *   Install: apt install libsymengine-dev  (Ubuntu)
 *            brew install symengine        (macOS)
 *   Or build from source: https://github.com/symengine/symengine
 *
 * JSON Interface:
 * {
 *   "command": "symbolic_diff",
 *   "expression": "x**2 + sin(x)",   // Expression to differentiate
 *   "variable": "x",                 // Variable to differentiate with respect to
 *   "order": 1                       // Order of derivative (optional, default 1)
 * }
 *
 * Response:
 * {
 *   "status": "success",
 *   "result": {
 *     "derivative": "2*x + cos(x)",
 *     "simplified": "2*x + cos(x)",
 *     "latex": "2x + \\cos(x)"
 *   },
 *   "stats": { "elapsed_ms": 1.2 }
 * }
 */

#include "../include/command.hpp"

#ifdef HAVE_SYMENGINE
#include <symengine/expression.h>
#include <symengine/symbol.h>
#include <symengine/derivative.h>
#include <symengine/simplify.h>
#include <symengine/latex.h>
#include <symengine/parser.h>

using namespace SymEngine;
#endif

class SymbolicDiffCommand : public Command {
public:
    std::string name() const override { return "symbolic_diff"; }
    std::string description() const override {
        return "Symbolic differentiation using SymEngine";
    }

    json execute(const json& params) override {
#ifndef HAVE_SYMENGINE
        return error("SymEngine not available. Install with: apt install libsymengine-dev");
#else
        Timer timer;

        // Parse parameters
        if (!params.contains("expression")) {
            return error("Missing 'expression' parameter");
        }

        std::string expr_str = params["expression"].get<std::string>();
        std::string var_name = params.value("variable", "x");
        int order = params.value("order", 1);

        if (order < 1 || order > 10) {
            return error("Derivative order must be between 1 and 10");
        }

        try {
            // Parse expression
            auto expr = parse(expr_str);
            auto var = symbol(var_name);

            // Compute derivative
            auto deriv = expr;
            for (int i = 0; i < order; i++) {
                deriv = diff(deriv, var);
            }

            // Simplify
            auto simplified = simplify(deriv);

            json result;
            result["derivative"] = deriv->__str__();
            result["simplified"] = simplified->__str__();
            result["latex"] = latex(*simplified);
            result["order"] = order;
            result["variable"] = var_name;

            json stats;
            stats["elapsed_ms"] = timer.elapsed_ms();

            return success(result, stats);
        } catch (const std::exception& e) {
            return error(std::string("Differentiation failed: ") + e.what());
        }
#endif
    }
};

// Factory function
std::unique_ptr<Command> create_symbolic_diff() {
    return std::make_unique<SymbolicDiffCommand>();
}
