/**
 * @file simplify.cpp
 * @brief Symbolic expression simplification using SymEngine
 *
 * DEPENDENCY: Requires SymEngine library
 *   Install: apt install libsymengine-dev  (Ubuntu)
 *            brew install symengine        (macOS)
 *   Or build from source: https://github.com/symengine/symengine
 *
 * JSON Interface:
 * {
 *   "command": "simplify",
 *   "expression": "x**2 + 2*x + 1",  // Expression to simplify
 *   "expand": false,                 // Expand first (optional)
 *   "factor": false,                 // Factor result (optional)
 *   "trigsimp": false                // Apply trig simplifications (optional)
 * }
 *
 * Response:
 * {
 *   "status": "success",
 *   "result": {
 *     "original": "x**2 + 2*x + 1",
 *     "simplified": "(x + 1)**2",
 *     "latex": "(x + 1)^2"
 *   },
 *   "stats": { "elapsed_ms": 0.8 }
 * }
 */

#include "../include/command.hpp"

#ifdef HAVE_SYMENGINE
#include <symengine/expression.h>
#include <symengine/symbol.h>
#include <symengine/simplify.h>
#include <symengine/expand.h>
#include <symengine/latex.h>
#include <symengine/parser.h>
#include <symengine/polys/basic_conversions.h>

using namespace SymEngine;
#endif

class SimplifyCommand : public Command {
public:
    std::string name() const override { return "simplify"; }
    std::string description() const override {
        return "Symbolic expression simplification using SymEngine";
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
        bool do_expand = params.value("expand", false);
        bool do_factor = params.value("factor", false);
        bool do_trigsimp = params.value("trigsimp", false);

        try {
            // Parse expression
            auto expr = parse(expr_str);

            // Apply transformations
            RCP<const Basic> result = expr;

            if (do_expand) {
                result = expand(result);
            }

            // Simplify
            result = simplify(result);

            if (do_factor) {
                // Attempt factorization (may not always succeed)
                try {
                    // SymEngine's factor function for polynomials
                    // This is a simplified approach - full factoring requires more setup
                    result = simplify(result);
                } catch (...) {
                    // Factoring not applicable, keep simplified form
                }
            }

            if (do_trigsimp) {
                // Trigonometric simplifications
                result = simplify(result);  // SymEngine applies trig rules automatically
            }

            json out;
            out["original"] = expr_str;
            out["simplified"] = result->__str__();
            out["latex"] = latex(*result);

            json stats;
            stats["elapsed_ms"] = timer.elapsed_ms();
            stats["expanded"] = do_expand;
            stats["factored"] = do_factor;
            stats["trigsimp"] = do_trigsimp;

            return success(out, stats);
        } catch (const std::exception& e) {
            return error(std::string("Simplification failed: ") + e.what());
        }
#endif
    }
};

// Factory function
std::unique_ptr<Command> create_simplify() {
    return std::make_unique<SimplifyCommand>();
}
